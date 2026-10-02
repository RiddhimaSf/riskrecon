-- 02_reconciliation.sql  |  Disagreement metrics and segment analysis (DuckDB dialect)
-- Run from the repo root:  duckdb < sql/02_reconciliation.sql
--   delta = DS - Actuarial ;  beyond tolerance when |delta| > max(0.35, 0.20 * mean score)

CREATE OR REPLACE TABLE members AS
SELECT * FROM read_csv_auto('data/synthetic_members.csv', header = true);

CREATE OR REPLACE VIEW recon AS
SELECT
    m.*,
    ds_risk_score - actuarial_risk_score                       AS delta,
    ABS(ds_risk_score - actuarial_risk_score)                  AS abs_delta,
    (ds_risk_score + actuarial_risk_score) / 2.0               AS mean_score,
    GREATEST(0.35, 0.20 * (ds_risk_score + actuarial_risk_score) / 2.0) AS tol,
    ABS(ds_risk_score - actuarial_risk_score)
        > GREATEST(0.35, 0.20 * (ds_risk_score + actuarial_risk_score) / 2.0) AS beyond_tolerance,
    1.2 * inpatient_visits + 0.5 * ed_visits + 0.08 * rx_count AS util_load
FROM members m;

-- Headline metrics
SELECT COUNT(*)                                   AS n_members,
       AVG(beyond_tolerance::INT)                 AS pct_beyond_tolerance,
       MEDIAN(delta)                              AS median_delta,
       AVG(abs_delta)                             AS mae,
       QUANTILE_CONT(abs_delta, 0.95)             AS p95_abs_delta
FROM recon;

-- Is disagreement random or structured?  By claims completeness
SELECT CASE WHEN claims_completeness < 0.70 THEN '1: <0.70'
            WHEN claims_completeness < 0.80 THEN '2: 0.70-0.80'
            WHEN claims_completeness < 0.90 THEN '3: 0.80-0.90'
            ELSE '4: >=0.90' END                  AS completeness_band,
       COUNT(*)                                   AS n,
       AVG(abs_delta)                             AS mean_abs_delta,
       AVG(beyond_tolerance::INT)                 AS pct_beyond_tolerance
FROM recon GROUP BY 1 ORDER BY 1;

-- By utilization
SELECT CASE WHEN util_load = 0 THEN '1: none'
            WHEN util_load <= 0.5 THEN '2: low'
            WHEN util_load <= 1.5 THEN '3: medium'
            ELSE '4: high' END                    AS utilization_band,
       COUNT(*)                                   AS n,
       AVG(abs_delta)                             AS mean_abs_delta,
       AVG(beyond_tolerance::INT)                 AS pct_beyond_tolerance
FROM recon GROUP BY 1 ORDER BY 1;

-- By enrollment duration
SELECT CASE WHEN months_enrolled <= 3 THEN '1: 1-3 mo'
            WHEN months_enrolled <= 6 THEN '2: 4-6 mo'
            WHEN months_enrolled <= 11 THEN '3: 7-11 mo'
            ELSE '4: 12 mo' END                   AS enrollment_band,
       COUNT(*)                                   AS n,
       AVG(abs_delta)                             AS mean_abs_delta,
       AVG(beyond_tolerance::INT)                 AS pct_beyond_tolerance
FROM recon GROUP BY 1 ORDER BY 1;
