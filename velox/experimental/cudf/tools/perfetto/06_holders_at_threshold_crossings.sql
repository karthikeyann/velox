-- Snapshots every non-zero operator and PlanNode holder when global live bytes
-- first cross the editable threshold. Each attribution view reconciles
-- independently to crossing_global_bytes.
-- Materialize the window-derived deltas once; otherwise SQLite may reevaluate
-- the vgm_samples view for every crossing-owner join.
DROP TABLE IF EXISTS vgm_threshold_owner_samples;
CREATE TABLE vgm_threshold_owner_samples AS
SELECT *
FROM vgm_samples
WHERE owner_kind IN ('plan', 'operator');
CREATE INDEX vgm_threshold_owner_samples_by_track_ts
ON vgm_threshold_owner_samples(track_id, ts);

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
    track_id AS global_track_id,
    track_name AS global_track_name,
    violation_id,
    MIN(ts) AS crossing_ts,
    MAX(ts + dur) - MIN(ts) AS violation_dur,
    MAX(value) AS violation_peak_bytes
  FROM numbered
  GROUP BY track_id, track_name, violation_id
),
annotated AS (
  SELECT
    v.*,
    (
      SELECT s.value
      FROM vgm_samples AS s
      WHERE
        s.track_id = v.global_track_id
        AND s.ts = v.crossing_ts
      ORDER BY s.id
      LIMIT 1
    ) AS crossing_global_bytes
  FROM violations AS v
),
owners AS (
  SELECT
    track_id,
    owner_kind,
    track_name
  FROM vgm_threshold_owner_samples
  GROUP BY track_id, owner_kind, track_name
),
candidates AS (
  SELECT
    v.*,
    o.owner_kind,
    o.track_id AS owner_track_id,
    o.track_name AS owner,
    COALESCE(
      (
        SELECT s.value
        FROM vgm_threshold_owner_samples AS s
        WHERE
          s.track_id = o.track_id
          AND s.ts <= v.crossing_ts
        ORDER BY s.ts DESC, s.id DESC
        LIMIT 1
      ),
      0
    ) AS held_bytes
  FROM annotated AS v
  CROSS JOIN owners AS o
),
at_crossing AS (
  SELECT *
  FROM candidates
  WHERE held_bytes > 0
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY global_track_id, violation_id, owner_kind
      ORDER BY held_bytes DESC, owner
    ) AS holder_rank,
    SUM(held_bytes) OVER (
      PARTITION BY global_track_id, violation_id, owner_kind
    ) AS attribution_total_bytes
  FROM at_crossing
)
SELECT
  r.crossing_ts AS ts,
  0 AS dur,
  printf(
    '#%d holds %.3f GiB at %.3f GiB threshold crossing',
    r.holder_rank,
    r.held_bytes / (1024.0 * 1024.0 * 1024.0),
    cfg.threshold_gib
  ) AS name,
  r.owner AS pivot,
  r.violation_id,
  r.violation_dur,
  r.owner_kind,
  r.holder_rank,
  r.owner_track_id,
  r.owner,
  r.held_bytes,
  100.0 * r.held_bytes / r.crossing_global_bytes
    AS percent_of_crossing_global,
  r.crossing_global_bytes,
  r.violation_peak_bytes,
  r.attribution_total_bytes,
  r.attribution_total_bytes - r.crossing_global_bytes
    AS reconciliation_delta_bytes
FROM ranked AS r
CROSS JOIN vgm_config AS cfg
ORDER BY r.crossing_ts, r.owner_kind, r.holder_rank;
