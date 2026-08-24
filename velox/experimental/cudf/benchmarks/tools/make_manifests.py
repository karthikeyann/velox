#!/usr/bin/env python3
"""Split an S3 object inventory into one manifest per NIC.

`velox_cudf_kvikio_read_benchmark` takes a manifest of object URIs. Running one
process per NIC needs one manifest per NIC, and they must be disjoint: pointed at
the same objects, every process after the first reads what the S3 front end has
already cached and reports a number that is not a transport measurement.

Objects smaller than the range size are dropped, so that every range GET is
full-sized and the geometry stays uniform.

Input is either a RAPIDS-style inventory JSON (an ``objects`` list of
``{key, contentLength}``) or the output of ``aws s3api list-objects-v2``
(``Contents`` of ``{Key, Size}``).

    ./make_manifests.py --inventory inv.json --bucket rapids-tpch \
        --out-dir /tmp/manifests --nics enp135s0,enp153s0,enp170s0,enp187s0
"""

import argparse
import json
import pathlib


def load_objects(path):
    doc = json.loads(pathlib.Path(path).read_text())
    if isinstance(doc, dict) and "objects" in doc:
        return [(o["key"], o["contentLength"]) for o in doc["objects"]]
    if isinstance(doc, dict) and "Contents" in doc:
        return [(o["Key"], o["Size"]) for o in doc["Contents"]]
    if isinstance(doc, dict) and "tasks" in doc:
        return [(t["key"], t["size"]) for t in doc["tasks"]]
    raise SystemExit("unrecognised inventory: want 'objects', 'Contents' or 'tasks'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nics", required=True, help="comma-separated interface names")
    ap.add_argument("--prefix", default="full", help="manifest filename prefix")
    ap.add_argument(
        "--min-object-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="drop objects below this, so every range GET is full-sized",
    )
    args = ap.parse_args()

    nics = [n for n in args.nics.split(",") if n]
    objects = [o for o in load_objects(args.inventory) if o[1] >= args.min_object_bytes]
    if len(objects) < len(nics):
        raise SystemExit(
            f"only {len(objects)} objects clear the size floor, need at least {len(nics)}"
        )

    # Largest-first greedy bin packing, so the slices carry near-equal bytes even
    # though the objects differ in size.
    bins = [[] for _ in nics]
    totals = [0] * len(nics)
    for key, size in sorted(objects, key=lambda kv: -kv[1]):
        i = min(range(len(nics)), key=lambda j: totals[j])
        bins[i].append(key)
        totals[i] += size

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for nic, keys, total in zip(nics, bins, totals):
        path = out / f"{args.prefix}.{nic}.manifest"
        with path.open("w") as f:
            f.write(f"# {nic}: {len(keys)} objects, {total} bytes\n")
            for key in sorted(keys):
                f.write(f"s3://{args.bucket}/{key}\n")
        print(f"{path}  {len(keys):4d} objects  {total / 2**30:8.1f} GiB")


if __name__ == "__main__":
    main()
