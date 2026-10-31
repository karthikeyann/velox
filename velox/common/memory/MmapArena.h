/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <unordered_set>

#include "velox/common/memory/MemoryAllocator.h"

namespace facebook::velox::memory {

class MmapArena {
 public:
  /// Single MmapArena capacity is determined by mmap_arena_capacity_ratio ratio
  /// but the capacity should be at least kMinCapacityBytes if the calculated
  /// capacity from the ratio is too small to serve large allocations.
  static constexpr int64_t kMinCapacityBytes = 128 * 1024 * 1024; // 128M

  /// MmapArena capacity should be multiple of kMinGrainSizeBytes.
  static constexpr uint64_t kMinGrainSizeBytes = 1024 * 1024; // 1M

  explicit MmapArena(size_t capacityBytes);
  ~MmapArena();

  void* allocate(uint64_t bytes);

  /// Returns the block of 'bytes' at 'address' to the arena. The first
  /// 'residentBytes' of the block, rounded up to a huge page boundary, stay
  /// mapped and are retained for reuse: a later allocation that gets these
  /// addresses uses them without page faults and sees their old contents.
  /// Retained memory elsewhere in the block is advised away.
  void free(void* address, uint64_t bytes, uint64_t residentBytes);

  /// Stops retaining the memory in [address, address + bytes) and returns how
  /// many of these bytes were retained. The caller uses this memory from now
  /// on, e.g. a new allocation or the growth of one.
  uint64_t takeRetained(void* address, uint64_t bytes);

  /// Advises away retained memory until at least 'targetBytes' are released or
  /// nothing is retained. Returns the number of bytes released.
  uint64_t releaseRetained(uint64_t targetBytes);

  /// Returns the number of bytes that free() retained and that are not taken
  /// or released since.
  uint64_t retainedBytes() const {
    return retainedBytes_;
  }

  void* address() const {
    return reinterpret_cast<void*>(address_);
  }

  uint64_t byteSize() const {
    return byteSize_;
  }

  const std::map<uintptr_t, uint64_t>& freeList() const {
    return freeList_;
  }

  const std::map<uint64_t, std::unordered_set<uintptr_t>>& freeLookup() const {
    return freeLookup_;
  }

  uint64_t freeBytes() {
    return freeBytes_;
  }

  bool empty() {
    return freeBytes_ == byteSize_;
  }

  /// Checks internal consistency of this MmapArena. Returns true if OK. May
  /// return false if there are concurrent allocations and frees during the
  /// consistency check. This is a false positive but not dangerous. This is for
  /// test only
  bool checkConsistency() const;

  /// translate lookup table to a string for debugging purpose only.
  std::string freeLookupStr() {
    std::stringstream lookupStr;
    for (auto itr = freeLookup_.begin(); itr != freeLookup_.end(); ++itr) {
      lookupStr << "\n{" << itr->first << "->[";
      for (auto itrInner = itr->second.begin(); itrInner != itr->second.end();
           itrInner++) {
        lookupStr << *itrInner << ", ";
      }
      lookupStr << "]}\n";
    }
    return lookupStr.str();
  }

  std::string toString() const;

 private:
  // Rounds up size to the next power of 2.
  static uint64_t roundBytes(uint64_t bytes);

  std::map<uintptr_t, uint64_t>::iterator addFreeBlock(
      uintptr_t addr,
      uint64_t bytes);

  void removeFromLookup(uintptr_t addr, uint64_t bytes);

  void removeFreeBlock(uintptr_t addr, uint64_t bytes);

  void removeFreeBlock(std::map<uintptr_t, uint64_t>::iterator& itr);

  // Adds [address, address + bytes), which must not overlap a retained range,
  // to 'retained_'.
  void addRetained(uintptr_t address, uint64_t bytes);

  // Removes [begin, end) from 'retained_' and returns the number of bytes
  // removed. Advises away the removed memory if 'adviseAway' is true.
  uint64_t removeRetained(uintptr_t begin, uintptr_t end, bool adviseAway);

  // Total capacity size of this arena.
  const uint64_t byteSize_;

  // Starting address of this arena.
  uint8_t* address_;

  std::atomic<uint64_t> freeBytes_;

  // A sorted list with each entry mapping from free block address to size of
  // the free block
  std::map<uintptr_t, uint64_t> freeList_;

  // A sorted look-up structure that stores the block size as key and a set of
  // addresses of that size as value.
  std::map<uint64_t, std::unordered_set<uintptr_t>> freeLookup_;

  // Mapped memory kept by free() for reuse, from start address to size. The
  // ranges do not overlap and adjacent ones are merged. A range lies in free
  // blocks or in the part of an allocated block beyond the size of its
  // allocation, e.g. the unused reserve of a growable contiguous allocation.
  std::map<uintptr_t, uint64_t> retained_;

  // Sum of the sizes in 'retained_'.
  uint64_t retainedBytes_{0};
};

/// A class that manages a set of MmapArenas. It is able to adapt itself by
/// growing the number of its managed MmapArena's when extreme memory
/// fragmentation happens.
class ManagedMmapArenas {
 public:
  ManagedMmapArenas(
      uint64_t singleArenaCapacity,
      std::function<void(void* address, size_t bytes)> onMap = nullptr);

  void* allocate(uint64_t bytes);

  /// Frees the block in the arena that contains 'address'. See
  /// MmapArena::free(). An empty arena other than the current one is unmapped,
  /// together with the memory it retained.
  void free(void* address, uint64_t bytes, uint64_t residentBytes);

  /// See MmapArena::takeRetained().
  uint64_t takeRetained(void* address, uint64_t bytes);

  /// See MmapArena::releaseRetained().
  uint64_t releaseRetained(uint64_t targetBytes);

  /// Returns the memory retained by all arenas.
  uint64_t retainedBytes() const;

  const std::map<uintptr_t, std::shared_ptr<MmapArena>>& arenas() const {
    return arenas_;
  }

 private:
  // Returns the arena that contains 'bytes' at 'address'.
  std::map<uintptr_t, std::shared_ptr<MmapArena>>::iterator findArena(
      void* address,
      uint64_t bytes);

  // Capacity in bytes for a single MmapArena managed by this.
  const uint64_t singleArenaCapacity_;

  // Invoked on each arena region as it is mapped.
  const std::function<void(void* address, size_t bytes)> onMap_;

  // A sorted list of MmapArena by its initial address
  std::map<uintptr_t, std::shared_ptr<MmapArena>> arenas_;

  // All allocations should come from this MmapArena. When it is no longer able
  // to handle allocations it will be updated to a newly created MmapArena.
  std::shared_ptr<MmapArena> currentArena_;
};

} // namespace facebook::velox::memory
