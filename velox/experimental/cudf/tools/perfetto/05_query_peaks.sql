-- Reconstructs exact per-query logical live bytes from the allocation-origin
-- operator deltas. This remains valid when multiple queries overlap.
WITH operator_deltas AS (
  SELECT
    ts,
    track_name AS owner,
    SUBSTR(
      track_name,
      INSTR(track_name, 'query=') + LENGTH('query='),
      INSTR(
        SUBSTR(track_name, INSTR(track_name, 'query=') + LENGTH('query=')),
        ' | task='
      ) - 1
    ) AS query_id,
    delta_bytes
  FROM vgm_samples
  WHERE owner_kind = 'operator'
),
query_deltas AS (
  SELECT
    query_id,
    ts,
    SUM(delta_bytes) AS delta_bytes
  FROM operator_deltas
  GROUP BY query_id, ts
),
query_live AS (
  SELECT
    query_id,
    ts,
    delta_bytes,
    SUM(delta_bytes) OVER (
      PARTITION BY query_id
      ORDER BY ts
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS live_bytes
  FROM query_deltas
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY query_id
      ORDER BY live_bytes DESC, ts
    ) AS peak_rank
  FROM query_live
)
SELECT
  r.ts,
  0 AS dur,
  printf(
    'Query peak %.3f GiB | trigger %+0.3f MiB',
    r.live_bytes / (1024.0 * 1024.0 * 1024.0),
    r.delta_bytes / (1024.0 * 1024.0)
  ) AS name,
  r.query_id AS pivot,
  r.query_id,
  r.live_bytes AS peak_bytes,
  r.delta_bytes AS trigger_delta_bytes,
  COALESCE(
    (
      SELECT o.owner
      FROM operator_deltas AS o
      WHERE o.query_id = r.query_id AND o.ts = r.ts
      ORDER BY o.delta_bytes DESC, o.owner
      LIMIT 1
    ),
    '<unattributed>'
  ) AS trigger_owner
FROM ranked AS r
WHERE r.peak_rank = 1
ORDER BY r.ts;
