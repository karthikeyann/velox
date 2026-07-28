-- Debug-slice compatible: ts, dur, name, and pivot are the first columns.
WITH peak AS (
  SELECT
    p.*,
    COALESCE(
      (
        SELECT o.track_name
        FROM vgm_samples AS o
        WHERE
          o.owner_kind = 'operator'
          AND o.ts = p.ts
          AND o.delta_bytes > 0
        ORDER BY o.delta_bytes DESC, o.track_id
        LIMIT 1
      ),
      '<unattributed>'
    ) AS trigger_owner
  FROM vgm_global_peak AS p
)
SELECT
  ts,
  0 AS dur,
  printf(
    'Global peak %.3f GiB | trigger +%.3f MiB | %s',
    value / (1024.0 * 1024.0 * 1024.0),
    delta_bytes / (1024.0 * 1024.0),
    trigger_owner
  ) AS name,
  track_name AS pivot,
  value AS peak_bytes,
  delta_bytes AS trigger_delta_bytes,
  trigger_owner
FROM peak
ORDER BY ts;
