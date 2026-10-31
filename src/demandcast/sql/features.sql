-- One row per real clock hour. Missing targets remain NULL, never zero-filled.
WITH aligned AS (
    SELECT g.ts, o.y
    FROM hourly_grid AS g
    LEFT JOIN observations AS o ON g.ts = o.ts
)
SELECT
    ts,
    y,
    date_trunc('day', ts) AS forecast_origin,
    lag(y, 24) OVER w AS lag_24,
    lag(y, 48) OVER w AS lag_48,
    lag(y, 168) OVER w AS lag_168,
    avg(y) OVER (ORDER BY ts ROWS BETWEEN 47 PRECEDING AND 24 PRECEDING) AS rolling_mean_24,
    avg(y) OVER (ORDER BY ts ROWS BETWEEN 191 PRECEDING AND 24 PRECEDING) AS rolling_mean_168,
    count(y) OVER (ORDER BY ts ROWS BETWEEN 47 PRECEDING AND 24 PRECEDING) AS history_count_24,
    count(y) OVER (ORDER BY ts ROWS BETWEEN 191 PRECEDING AND 24 PRECEDING) AS history_count_168
FROM aligned
WINDOW w AS (ORDER BY ts)
ORDER BY ts
