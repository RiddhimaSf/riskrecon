import numpy as np
import pandas as pd

from src import validation as V
from src.synthetic import generate


def members(n=1500):
    return generate(n, seed=11)[0]


def status(result):
    return result.status


def test_clean_synthetic_table_passes_integrity_gates():
    m = members()
    for g in (V.check_unique_ids, V.check_schema, V.check_freshness, V.check_score_ranges, V.check_missingness):
        assert status(g(m)) == "PASS", g.__name__


def test_duplicate_member_ids_fail():
    m = members(); m = pd.concat([m, m.iloc[:3]])
    r = V.check_unique_ids(m)
    assert r.status == "FAIL" and r.value == 3


def test_missing_column_fails_schema_gate():
    assert V.check_schema(members().drop(columns=["claims_completeness"])).status == "FAIL"


def test_out_of_range_scores_fail():
    m = members(); m.loc[m.index[0], "ds_risk_score"] = -1.0
    assert V.check_score_ranges(m).status == "FAIL"


def test_future_dates_fail_freshness():
    m = members(); m["ds_snapshot_date"] = pd.to_datetime(m["ds_snapshot_date"])
    m.loc[m.index[0], "ds_snapshot_date"] = pd.Timestamp("2026-06-01")
    assert V.check_freshness(m).status == "FAIL"


def test_duplicate_claim_lines_are_detected():
    m = members(); m.loc[m.index[:50], "claim_lines_total"] = m.loc[m.index[:50], "claim_lines_distinct"] * 2 + 2
    assert V.check_duplicates(m).status == "FAIL"


def test_psi_is_zero_for_identical_and_large_for_shifted():
    x = np.random.default_rng(0).normal(2, 1, 5000)
    assert V.psi(x, x) < 1e-6
    assert V.psi(x, x + 1.5) > 0.25
