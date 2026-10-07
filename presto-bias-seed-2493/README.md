Persisted repro from the Presto Bias Fuzzer run on facebookincubator/velox#19353
(GitHub Actions run 37526938387, seed 2493, `--assign_function_tickets between=20`).

Replay, after adding `functions::prestosql::registerInternalFunctions();` next to
`registerAllScalarFunctions()` in velox/expression/tests/ExpressionRunnerTest.cpp:

    velox_expression_runner_test \
      --input_paths presto-bias-seed-2493/input_vector_0 \
      --input_selectivity_vector_paths presto-bias-seed-2493/input_selectivity_vector_0 \
      --sql_path presto-bias-seed-2493/sql \
      --complex_constant_path presto-bias-seed-2493/complex_constants \
      --result_path presto-bias-seed-2493/result_vector \
      --input_row_metadata_path presto-bias-seed-2493/input_row_metadata \
      --mode verify
