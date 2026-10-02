-- 03_investigation_queue.sql  |  Priority score and the investigation queue (DuckDB dialect)
-- Input: data/reconciliation_results.csv  (written by `python -m src.pipeline`; it carries the
--        deterministic root-cause label and confidence from src/diagnostics.py)
--
--   priority = 1 - exp( -|delta| * impact * confidence / K ),  K = 1.0
--   impact   = 0.6 * [scores fall in different decision bands] + 0.4 * min(1, max(score) / 6)
--   bands    = [<2, 2-4, 4-6, 6+]   (hypothetical, for illustration)

CREATE OR REPLACE TABLE results AS
SELECT * FROM read_csv_auto('data/reconciliation_results.csv', header = true);

CREATE OR REPLACE VIEW scored AS
SELECT *,
       (  (ds_risk_score >= 2)::INT + (ds_risk_score >= 4)::INT + (ds_risk_score >= 6)::INT
        <> (actuarial_risk_score >= 2)::INT + (actuarial_risk_score >= 4)::INT + (actuarial_risk_score >= 6)::INT
       )::INT * 0.6
       + 0.4 * LEAST(1.0, GREATEST(ds_risk_score, actuarial_risk_score) / 6.0)   AS impact_sql
FROM results;

CREATE OR REPLACE VIEW investigation_queue AS
SELECT ROW_NUMBER() OVER (ORDER BY priority_sql DESC, member_id) AS rank,
       member_id, ds_risk_score, actuarial_risk_score, delta, cause, evidence,
       impact_sql AS impact, confidence, priority_sql AS priority
FROM (
    SELECT *, 1 - EXP(-abs_delta * impact_sql * confidence / 1.0) AS priority_sql
    FROM scored
    WHERE beyond_tolerance
)
ORDER BY priority_sql DESC, member_id
LIMIT 50;

SELECT * FROM investigation_queue;
