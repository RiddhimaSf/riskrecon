-- 01_validation.sql  |  Data gates, expressed as SQL (DuckDB dialect)
-- Run from the repo root:  duckdb < sql/01_validation.sql
-- Each row is one gate. status = PASS when value is within the limit.

CREATE OR REPLACE TABLE members AS
SELECT * FROM read_csv_auto('data/synthetic_members.csv', header = true);

SELECT * FROM (
    -- unique member IDs
    SELECT 'unique_member_ids' AS gate,
           (COUNT(*) - COUNT(DISTINCT member_id))::DOUBLE AS value,
           '== 0' AS "limit",
           CASE WHEN COUNT(*) = COUNT(DISTINCT member_id) THEN 'PASS' ELSE 'FAIL' END AS status
    FROM members

    UNION ALL
    -- acceptable missingness in required scoring fields (last_claim_date may be null: no claims)
    SELECT 'acceptable_missingness',
           AVG((member_id IS NULL OR ds_risk_score IS NULL OR actuarial_risk_score IS NULL
                OR claims_completeness IS NULL OR months_enrolled IS NULL
                OR ds_snapshot_date IS NULL OR act_snapshot_date IS NULL)::INT),
           '<= 0.01',
           CASE WHEN AVG((member_id IS NULL OR ds_risk_score IS NULL OR actuarial_risk_score IS NULL
                OR claims_completeness IS NULL OR months_enrolled IS NULL
                OR ds_snapshot_date IS NULL OR act_snapshot_date IS NULL)::INT) <= 0.01
                THEN 'PASS' ELSE 'FAIL' END
    FROM members

    UNION ALL
    -- claims completeness (WARN-level gate)
    SELECT 'claims_completeness',
           AVG((claims_completeness < 0.80)::INT),
           '<= 0.05',
           CASE WHEN AVG((claims_completeness < 0.80)::INT) <= 0.05 THEN 'PASS' ELSE 'WARN' END
    FROM members

    UNION ALL
    -- date freshness: nothing dated after the reference snapshot
    SELECT 'date_freshness',
           AVG((last_claim_date > DATE '2025-12-31'
                OR ds_snapshot_date > DATE '2025-12-31'
                OR act_snapshot_date > DATE '2025-12-31')::INT),
           '<= 0',
           CASE WHEN AVG((last_claim_date > DATE '2025-12-31'
                OR ds_snapshot_date > DATE '2025-12-31'
                OR act_snapshot_date > DATE '2025-12-31')::INT) <= 0 THEN 'PASS' ELSE 'FAIL' END
    FROM members

    UNION ALL
    -- duplicate detection: members whose claim lines are >= 8% duplicates
    SELECT 'duplicate_detection',
           AVG((COALESCE(1 - claim_lines_distinct::DOUBLE / NULLIF(claim_lines_total, 0), 0) >= 0.08)::INT),
           '<= 0.005',
           CASE WHEN AVG((COALESCE(1 - claim_lines_distinct::DOUBLE / NULLIF(claim_lines_total, 0), 0) >= 0.08)::INT)
                     <= 0.005 THEN 'PASS' ELSE 'FAIL' END
    FROM members

    UNION ALL
    -- score ranges
    SELECT 'score_ranges',
           AVG((ds_risk_score NOT BETWEEN 0 AND 15 OR actuarial_risk_score NOT BETWEEN 0 AND 15
                OR ds_risk_score IS NULL OR actuarial_risk_score IS NULL)::INT),
           '== 0',
           CASE WHEN AVG((ds_risk_score NOT BETWEEN 0 AND 15 OR actuarial_risk_score NOT BETWEEN 0 AND 15
                OR ds_risk_score IS NULL OR actuarial_risk_score IS NULL)::INT) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM members
) ORDER BY gate;
