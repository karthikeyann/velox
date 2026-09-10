# cuDF dependency patch

`jit-warp-validity.patch` fixes null-aware JIT transforms in pinned cuDF
`5beaa5954688fcb12236ffb434e192ea2c77db30`. The per-lane grid-stride loop and
`__activemask()` can split a warp's preceding iteration when lanes reach a
partial final iteration, corrupting its output validity word. The patch keeps
the null-aware loop warp-uniform and ballots all lanes before guarded loads.
The non-null-aware path is unchanged.

This was found during PRESTO SF1000 validation. The Velox regression test is
`CudfFilterProjectTest.jitNullPredicatesAcrossPartialWarpIterations`; it compares
large nullable projections and filters with CPU Velox. For a system cuDF build,
apply the same patch and rebuild its embedded JIT sources. The normal bundled
dependency build applies it automatically. Remove the patch when upgrading to
a cuDF revision with the equivalent fix.
