-- Produces one merged slice for each interval where global live bytes are
-- strictly greater than the editable analysis-time threshold.
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
    i.owner_kind = 'global'
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
    track_id,
    track_name,
    violation_id,
    MIN(ts) AS ts,
    MAX(ts + dur) - MIN(ts) AS dur,
    MAX(value) AS peak_bytes
  FROM numbered
  GROUP BY track_id, track_name, violation_id
),
annotated AS (
  SELECT
    v.*,
    (
      SELECT s.delta_bytes
      FROM vgm_samples AS s
      WHERE s.track_id = v.track_id AND s.ts = v.ts
      ORDER BY s.id
      LIMIT 1
    ) AS crossing_delta_bytes,
    COALESCE(
      (
        SELECT o.track_name
        FROM vgm_samples AS o
        WHERE
          o.owner_kind = 'operator'
          AND o.ts = v.ts
          AND o.delta_bytes > 0
        ORDER BY o.delta_bytes DESC, o.track_id
        LIMIT 1
      ),
      '<unattributed>'
    ) AS crossing_owner
  FROM violations AS v
)
SELECT
  a.ts,
  a.dur,
  printf(
    'Global > %.3f GiB | peak %.3f GiB | crossing +%.3f MiB | %s',
    cfg.threshold_gib,
    a.peak_bytes / (1024.0 * 1024.0 * 1024.0),
    a.crossing_delta_bytes / (1024.0 * 1024.0),
    a.crossing_owner
  ) AS name,
  a.track_name AS pivot,
  a.peak_bytes,
  a.crossing_delta_bytes,
  a.crossing_owner
FROM annotated AS a
CROSS JOIN vgm_config AS cfg
ORDER BY a.ts;
