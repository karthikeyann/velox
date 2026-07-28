INCLUDE PERFETTO MODULE counters.intervals;

-- Analysis policy: edit only this GiB value, then rerun this file.
-- Values equal to the threshold are not violations; the queries use `>`.
DROP VIEW IF EXISTS vgm_config;
CREATE VIEW vgm_config AS
SELECT
  8.0 AS threshold_gib,
  8.0 * 1024.0 * 1024.0 * 1024.0 AS threshold_bytes;

DROP VIEW IF EXISTS vgm_tracks;
CREATE VIEW vgm_tracks AS
SELECT
  id AS track_id,
  name AS track_name,
  CASE
    WHEN name GLOB 'Overall RMM logical live bytes*' THEN 'global'
    WHEN name GLOB 'PlanNode RMM logical live bytes | *' THEN 'plan'
    WHEN name GLOB 'Operator RMM logical live bytes | *' THEN 'operator'
  END AS owner_kind
FROM counter_track
WHERE
  name GLOB 'Overall RMM logical live bytes*'
  OR name GLOB 'PlanNode RMM logical live bytes | *'
  OR name GLOB 'Operator RMM logical live bytes | *';

-- delta_bytes is the allocation/free transition represented by each sample.
-- Capture must assign strictly increasing timestamps to global transitions.
DROP VIEW IF EXISTS vgm_samples;
CREATE VIEW vgm_samples AS
SELECT
  c.id,
  c.ts,
  c.track_id,
  c.value,
  t.track_name,
  t.owner_kind,
  c.value - COALESCE(
    LAG(c.value) OVER (
      PARTITION BY c.track_id
      ORDER BY c.ts, c.id
    ),
    0
  ) AS delta_bytes
FROM counter AS c
JOIN vgm_tracks AS t ON t.track_id = c.track_id;

-- Treat each counter sample as a step value until the next sample.
DROP VIEW IF EXISTS vgm_intervals;
CREATE VIEW vgm_intervals AS
SELECT
  i.id,
  i.ts,
  i.dur,
  i.track_id,
  i.value,
  i.next_value,
  i.delta_value,
  t.track_name,
  t.owner_kind
FROM counter_leading_intervals!((
  SELECT id, ts, track_id, value
  FROM vgm_samples
)) AS i
JOIN vgm_tracks AS t ON t.track_id = i.track_id;

-- Choose the first occurrence if the same maximum is reached more than once.
DROP VIEW IF EXISTS vgm_global_peak;
CREATE VIEW vgm_global_peak AS
WITH ranked AS (
  SELECT
    s.*,
    ROW_NUMBER() OVER (
      PARTITION BY s.track_id
      ORDER BY s.value DESC, s.ts, s.id
    ) AS peak_rank
  FROM vgm_samples AS s
  WHERE s.owner_kind = 'global'
)
SELECT *
FROM ranked
WHERE peak_rank = 1;
