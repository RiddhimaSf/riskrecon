import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.reconciliation import (add_delta, tolerance, headline_metrics, impact, priority_score,
                                investigation_queue, completeness_lift)
from src.synthetic import generate


def toy(ds, act, **kw):
    n = len(ds)
    base = dict(member_id=[f"M{i}" for i in range(n)], ds_risk_score=ds, actuarial_risk_score=act,
                claims_completeness=[1.0] * n)
    base.update(kw)
    return pd.DataFrame(base)


def test_delta_is_ds_minus_actuarial():
    d = add_delta(toy([3.0, 1.0], [2.0, 2.5]))
    assert d["delta"].tolist() == [1.0, -1.5]


def test_tolerance_is_larger_of_absolute_and_relative():
    # low scores -> absolute floor; high scores -> relative band
    assert tolerance(1.0) == pytest.approx(C.ABS_TOL)
    assert tolerance(10.0) == pytest.approx(C.REL_TOL * 10.0)


def test_beyond_tolerance_boundary():
    d = add_delta(toy([2.0 + 0.30, 2.0 + 0.50], [2.0, 2.0]))
    assert d["beyond_tolerance"].tolist() == [False, True]


def test_headline_metrics_on_toy_data():
    d = add_delta(toy([2.0, 2.0, 2.0, 4.0], [2.0, 2.1, 2.0, 2.0]))
    m = headline_metrics(d)
    assert m["n_beyond_tolerance"] == 1 and m["pct_beyond_tolerance"] == pytest.approx(0.25)
    assert m["mae"] == pytest.approx((0 + 0.1 + 0 + 2.0) / 4)
    assert m["median_delta"] == pytest.approx(0.0)


def test_impact_increases_when_scores_cross_a_decision_band():
    same_band = impact([2.2], [3.8])[0]
    crosses = impact([3.8], [4.2])[0]
    assert crosses > same_band


def test_priority_is_monotonic_in_each_factor_and_bounded():
    base = priority_score(1.0, 0.5, 0.5)
    assert priority_score(2.0, 0.5, 0.5) > base
    assert priority_score(1.0, 0.9, 0.5) > base
    assert priority_score(1.0, 0.5, 0.9) > base
    assert 0 <= priority_score(0.0, 0.5, 0.5) < priority_score(50, 1, 1) < 1.0000001


def test_zero_confidence_means_zero_priority():
    assert priority_score(3.0, 1.0, 0.0) == pytest.approx(0.0)


def test_generation_is_deterministic():
    a, _ = generate(2000, seed=3); b, _ = generate(2000, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_queue_is_sorted_flagged_only_and_sized(small_run):
    q = small_run["queue"]
    assert len(q) == C.QUEUE_SIZE
    assert q["priority"].is_monotonic_decreasing
    flagged = set(small_run["classified"].query("beyond_tolerance").member_id)
    assert set(q["member_id"]) <= flagged
    assert q["rank"].tolist() == list(range(1, len(q) + 1))


def test_incomplete_claims_are_structurally_riskier(small_run):
    assert completeness_lift(small_run["classified"])["lift"] > 2.0


def test_framework_recovers_planted_causes(small_run):
    rec = small_run["recovery_metrics"].set_index("cause")
    assert small_run["metrics"]["recovery_accuracy"] > 0.78
    for cause in ["SNAPSHOT_MISMATCH", "DUPLICATE_INPUT", "MODEL_VERSION"]:
        assert rec.loc[cause, "recall"] > 0.85, cause
    # the control: shocks with clean diagnostics must stay visible as UNEXPLAINED
    assert rec.loc["UNEXPLAINED", "recall"] > 0.90
