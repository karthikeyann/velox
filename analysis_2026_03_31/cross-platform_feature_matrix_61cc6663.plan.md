---
name: Cross-Platform Feature Matrix
overview: Create a comprehensive "Feature Maturity Matrix" HTML file comparing Presto Native (CPU), Presto GPU, Velox, Gluten (CPU), and Gluten GPU across all feature categories, using the same visual style as presto-native-gpu-report.html.
todos:
  - id: create-html
    content: Create the cross-platform-feature-matrix.html file with all 5 tabs, legend, and ~100+ feature rows across all categories
    status: completed
isProject: false
---

# Cross-Platform Feature Maturity Matrix

## Output

A single new HTML file at `/home/knataraj/AGENT/cross-platform-feature-matrix.html` styled identically to `[presto-native-gpu-report.html](AGENT/presto-native-gpu-report.html)` (dark theme, DM Serif Display + IBM Plex Sans fonts, badge-based status indicators, tabbed sections).

## Structure

### Legend (top of page, always visible)

Status badges with color coding:

- **Production** (green) -- battle-tested at scale, used in production deployments
- **Stable** (blue) -- complete, well-tested, ready for use
- **Active Dev** (amber) -- functional but still evolving
- **Experimental** (purple) -- partial or prototype
- **Not Supported** (red) -- not implemented
- **N/A** (gray) -- not applicable to this platform

### Tabbed Sections

**Tab 1: Operators** (~35 rows)
Columns: Operator | Presto Native | Presto GPU | Velox | Gluten | Gluten GPU

All operators derived from source analysis:

- **Scan**: TableScan, FileSourceScan, BatchScan
- **Filter/Project**: Filter, Project, FilterProject
- **Joins**: HashJoin, MergeJoin, SemiJoin, IndexJoin, SpatialJoin, NestedLoopJoin, BroadcastHashJoin, ShuffledHashJoin, SortMergeJoin, CartesianProduct
- **Aggregation**: HashAggregation, StreamingAggregation
- **Sort/Limit**: OrderBy, TopN, Limit, DistinctLimit
- **Window**: Window, RowNumber, TopNRowNumber, WindowGroupLimit
- **Misc**: Unnest/Generate, Expand/GroupId, MarkDistinct, AssignUniqueId, EnforceSingleRow, Union, Sample
- **Write**: TableWriter, DeleteNode
- **Exchange**: HTTP Exchange, LocalPartition, Shuffle, BroadcastExchange

Sources:

- Presto Native: 29 plan node types from `PrestoToVeloxQueryPlan.cpp`
- Velox CPU: ~40 operator types from `OperatorType.h`
- Velox cuDF: 8 compute operators + 2 infra operators from `experimental/cudf/exec/`
- Gluten: ~25 ExecTransformer classes from `gluten-substrait/` and `backends-velox/`

**Tab 2: Connectors and File Formats** (~20 rows)
Columns: Feature | Presto Native | Presto GPU | Velox | Gluten | Gluten GPU

- Hive Connector (Parquet/ORC/DWRF/Text)
- Iceberg, Delta Lake, Hudi, Paimon
- TPC-H, TPC-DS
- Arrow Flight, System connector
- Storage: S3, GCS, HDFS, ABFS
- File format read/write capabilities per format

**Tab 3: Execution and Infrastructure** (~20 rows)
Columns: Feature | Presto Native | Presto GPU | Velox | Gluten | Gluten GPU

- Vectorized execution, Memory arbitration, Spill-to-disk
- Dynamic filtering, SSD cache, Query trace/replay
- Shuffle services (Celeborn, Uniffle), Remote functions
- Fault tolerance, Sidecar optimizer
- GPU-specific: DriverAdapter, CudfPlanValidator/Checker, GPU session config, CPU fallback

**Tab 4: Functions and Expressions** (~15 rows)
Columns: Category | Presto Native | Presto GPU | Velox | Gluten | Gluten GPU

- Math/Arithmetic (~40+ functions)
- String (~60+ functions)
- Date/Time (~40+ functions)
- Comparison/Logic (~20+)
- Array, Map, JSON, URL, Geospatial, HLL/Sketch
- Aggregate functions (~30+)
- Window functions (~10+)
- Lambda/higher-order functions
- Cast framework, Decimal arithmetic
- Spark-specific functions (~234 registrations)

**Tab 5: Memory and Performance** (~15 rows)
Columns: Feature | Presto Native | Presto GPU | Velox | Gluten | Gluten GPU

- MemoryPool hierarchy, Memory arbitration, GPU memory (RMM)
- Spill framework (per operator), GPU spill
- Arrow bridge, Vector encodings
- SSD cache, Prefetch/preload
- Expression JIT, Filter reordering
- Multi-GPU support, GPU-aware scheduling

## Data Sources (verified from code)

- **Velox CPU operators**: `velox/exec/OperatorType.h` -- 40 operator types
- **Velox cuDF operators**: `velox/experimental/cudf/exec/Cudf*.{h,cpp}` -- 8 compute + 2 infra
- **Velox cuDF connectors**: `velox/experimental/cudf/connectors/hive/` -- Parquet read only
- **Presto Native plan nodes**: `PrestoToVeloxQueryPlan.cpp` -- 29 node types
- **Presto GPU integration**: `PRESTO_ENABLE_CUDF` flag, `CudfHiveConnectorFactory`, `cudf.`* config keys
- **Gluten operators**: 25+ `*ExecTransformer.scala` classes
- **Gluten GPU**: `CudfPlanValidator`, GPU shuffle writer/reader, `GpuBufferBatchResizer`, GPU CI

## Implementation Notes

- Use the exact CSS from `presto-native-gpu-report.html` (dark theme, sticky nav, badges, cards)
- Tabbed interface with JavaScript tab switching (same pattern as existing reports)
- Legend section above tabs, always visible
- Sticky column headers for tables
- Responsive layout for smaller screens
- Footer with generation date and source repos

