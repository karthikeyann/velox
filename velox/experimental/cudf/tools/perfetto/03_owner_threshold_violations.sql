-- Produces only PlanNode/operator instances whose own live bytes exceed the
-- editable threshold. Pivot on `pivot` to get one debug lane per offender.
WITH high_steps AS (
  SELECT
    i.*,
    CASE
      WHEN LAG(i.ts + i.dur) OVER (
        PARTITION BY i.track_id
        ORDER BY i.ts, i.id
      ) = i.ts THEN 0
      ELSE 1
    END AS starts_violation
  FROM vgm_intervals AS i
  CROSS JOIN vgm_config AS cfg
  WHERE
    i.owner_kind IN ('plan', 'operator')
    AND i.value > cfg.threshold_bytes
    AND i.dur >= 0
),
numbered AS (
  SELECT
    *,
    SUM(starts_violation) OVER (
      PARTITION BY track_id
      ORDER BY ts, id
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS violation_id
  FROM high_steps
),
violations AS (
  SELECT
    owner_kind,
    track_id,
    track_name,
    violation_id,
    MIN(ts) AS ts,
    MAX(ts + dur) - MIN(ts) AS dur,
    MAX(value) AS peak_bytes
  FROM numbered
  GROUP BY owner_kind, track_id, track_name, violation_id
)
SELECT
  v.ts,
  v.dur,
  printf(
    '%s > %.3f GiB | peak %.3f GiB',
    v.owner_kind,
    cfg.threshold_gib,
    v.peak_bytes / (1024.0 * 1024.0 * 1024.0)
  ) AS name,
  v.track_name AS pivot,
  v.owner_kind,
  v.track_name AS owner,
  v.peak_bytes,
  (
    SELECT s.delta_bytes
    FROM vgm_samples AS s
    WHERE s.track_id = v.track_id AND s.ts = v.ts
    ORDER BY s.id
    LIMIT 1
  ) AS crossing_delta_bytes
FROM violations AS v
CROSS JOIN vgm_config AS cfg
ORDER BY v.owner_kind, v.peak_bytes DESC, v.ts;
