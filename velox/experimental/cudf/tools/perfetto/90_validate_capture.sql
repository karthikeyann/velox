-- All rows should report PASS. Delta reconciliation proves that every global
-- transition has one complete operator view and one complete PlanNode view.
WITH
global_deltas AS (
  SELECT ts, delta_bytes
  FROM vgm_samples
  WHERE owner_kind = 'global'
),
operator_deltas AS (
  SELECT ts, SUM(delta_bytes) AS delta_bytes
  FROM vgm_samples
  WHERE owner_kind = 'operator'
  GROUP BY ts
),
plan_deltas AS (
  SELECT ts, SUM(delta_bytes) AS delta_bytes
  FROM vgm_samples
  WHERE owner_kind = 'plan'
  GROUP BY ts
),
operator_mismatches AS (
  SELECT g.ts
  FROM global_deltas AS g
  LEFT JOIN operator_deltas AS o USING (ts)
  WHERE ABS(g.delta_bytes - COALESCE(o.delta_bytes, 0)) > 0.5
),
plan_mismatches AS (
  SELECT g.ts
  FROM global_deltas AS g
  LEFT JOIN plan_deltas AS p USING (ts)
  WHERE ABS(g.delta_bytes - COALESCE(p.delta_bytes, 0)) > 0.5
),
duplicate_global_timestamps AS (
  SELECT ts
  FROM vgm_samples
  WHERE owner_kind = 'global'
  GROUP BY ts
  HAVING COUNT(*) > 1
),
perfetto_integrity_failures AS (
  SELECT COALESCE(SUM(ABS(value)), 0) AS actual
  FROM stats
  WHERE severity IN ('data_loss', 'error') AND value != 0
),
source_data_loss_markers AS (
  SELECT COUNT(*) AS actual
  FROM slice
  WHERE name = 'GPU memory trace data loss'
),
latest_samples AS (
  SELECT
    owner_kind,
    value,
    ROW_NUMBER() OVER (
      PARTITION BY track_id
      ORDER BY ts DESC, id DESC
    ) AS sample_rank
  FROM vgm_samples
),
final_totals AS (
  SELECT
    owner_kind,
    COALESCE(SUM(value), 0) AS actual
  FROM latest_samples
  WHERE sample_rank = 1
  GROUP BY owner_kind
),
final_values AS (
  SELECT
    COALESCE(
      (SELECT actual FROM final_totals WHERE owner_kind = 'global'),
      0
    ) AS global_bytes,
    COALESCE(
      (SELECT actual FROM final_totals WHERE owner_kind = 'plan'),
      0
    ) AS plan_bytes,
    COALESCE(
      (SELECT actual FROM final_totals WHERE owner_kind = 'operator'),
      0
    ) AS operator_bytes
),
checks AS (
  SELECT
    'exactly one global track' AS check_name,
    COUNT(*) AS actual,
    CASE WHEN COUNT(*) = 1 THEN 'PASS' ELSE 'FAIL' END AS status
  FROM vgm_tracks
  WHERE owner_kind = 'global'
  UNION ALL
  SELECT
    'at least one operator track',
    COUNT(*),
    CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END
  FROM vgm_tracks
  WHERE owner_kind = 'operator'
  UNION ALL
  SELECT
    'at least one PlanNode track',
    COUNT(*),
    CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END
  FROM vgm_tracks
  WHERE owner_kind = 'plan'
  UNION ALL
  SELECT
    'duplicate global timestamps',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
  FROM duplicate_global_timestamps
  UNION ALL
  SELECT
    'negative live-byte samples',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
  FROM vgm_samples
  WHERE value < 0
  UNION ALL
  SELECT
    'operator delta mismatches',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
  FROM operator_mismatches
  UNION ALL
  SELECT
    'PlanNode delta mismatches',
    COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
  FROM plan_mismatches
  UNION ALL
  SELECT
    'Perfetto data-loss or error stats',
    actual,
    CASE WHEN actual = 0 THEN 'PASS' ELSE 'FAIL' END
  FROM perfetto_integrity_failures
  UNION ALL
  SELECT
    'source-ledger data-loss markers',
    actual,
    CASE WHEN actual = 0 THEN 'PASS' ELSE 'FAIL' END
  FROM source_data_loss_markers
  UNION ALL
  SELECT
    'final global logical live bytes',
    global_bytes,
    CASE WHEN ABS(global_bytes) <= 0.5 THEN 'PASS' ELSE 'FAIL' END
  FROM final_values
  UNION ALL
  SELECT
    'final PlanNode aggregate mismatch',
    ABS(plan_bytes - global_bytes),
    CASE
      WHEN ABS(plan_bytes - global_bytes) <= 0.5 THEN 'PASS'
      ELSE 'FAIL'
    END
  FROM final_values
  UNION ALL
  SELECT
    'final operator aggregate mismatch',
    ABS(operator_bytes - global_bytes),
    CASE
      WHEN ABS(operator_bytes - global_bytes) <= 0.5 THEN 'PASS'
      ELSE 'FAIL'
    END
  FROM final_values
)
SELECT status, check_name, actual
FROM checks
ORDER BY status, check_name;
