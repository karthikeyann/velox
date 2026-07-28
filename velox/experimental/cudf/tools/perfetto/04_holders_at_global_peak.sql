-- Returns both attribution views. Filter owner_kind to 'operator' or 'plan'
-- when only one view is wanted. Each view should reconcile independently.
WITH candidates AS (
  SELECT
    p.track_id AS global_track_id,
    p.track_name AS global_track_name,
    p.ts AS peak_ts,
    p.value AS global_peak_bytes,
    s.owner_kind,
    s.track_id AS owner_track_id,
    s.track_name AS owner,
    s.ts AS owner_sample_ts,
    s.value AS held_bytes,
    ROW_NUMBER() OVER (
      PARTITION BY p.track_id, s.track_id
      ORDER BY s.ts DESC, s.id DESC
    ) AS latest_sample_rank
  FROM vgm_global_peak AS p
  JOIN vgm_samples AS s
    ON s.owner_kind IN ('plan', 'operator')
    AND s.ts <= p.ts
),
at_peak AS (
  SELECT *
  FROM candidates
  WHERE latest_sample_rank = 1 AND held_bytes > 0
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY global_track_id, owner_kind
      ORDER BY held_bytes DESC, owner
    ) AS holder_rank,
    SUM(held_bytes) OVER (
      PARTITION BY global_track_id, owner_kind
    ) AS attribution_total_bytes
  FROM at_peak
)
SELECT
  peak_ts AS ts,
  0 AS dur,
  printf(
    '#%d holds %.3f GiB at global peak %.3f GiB',
    holder_rank,
    held_bytes / (1024.0 * 1024.0 * 1024.0),
    global_peak_bytes / (1024.0 * 1024.0 * 1024.0)
  ) AS name,
  owner AS pivot,
  owner_kind,
  holder_rank,
  owner_track_id,
  owner,
  held_bytes,
  100.0 * held_bytes / global_peak_bytes AS percent_of_global_peak,
  global_peak_bytes,
  attribution_total_bytes,
  attribution_total_bytes - global_peak_bytes AS reconciliation_delta_bytes
FROM ranked
ORDER BY owner_kind, holder_rank;
