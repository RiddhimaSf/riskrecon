import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.copilot import evidence_packet, deterministic_summary, validate_explanation
from src.diagnostics import add_features, classify, fit_baseline, fit_sensitivity
from src.synthetic import generate


@pytest.fixture(scope="module")
def ctx():
    m, _ = generate(8000, seed=5)
    f = add_features(m)
    sens, base = fit_sensitivity(f), fit_baseline(f)
    clean = f[(f.claims_completeness >= 0.95) & (f.snapshot_gap_days == 0) & (~f.version_mismatch)
              & (f.dup_claim_ratio < 0.02) & f.load.between(0.9, 1.6) & (f.months_enrolled == 12)]
    row = m.loc[clean.index[0]]
    return m, sens, base, row


def case(ctx, **over):
    m, sens, base, row = ctx
    r = row.copy()
    for k, v in over.items():
        r[k] = v
    df = pd.DataFrame([r])
    for c in ["last_claim_date", "ds_snapshot_date", "act_snapshot_date"]:
        df[c] = pd.to_datetime(df[c])
    df["ds_risk_score"] = df["ds_risk_score"].astype(float)
    return classify(add_features(df), sens, base).iloc[0]


def scores(act, delta):
    return dict(actuarial_risk_score=act, ds_risk_score=act + delta)


def test_within_tolerance_is_not_classified(ctx):
    assert case(ctx, **scores(3.0, 0.1)).cause == "WITHIN_TOLERANCE"


def test_incomplete_history_with_lower_ds_score_is_data_completeness(ctx):
    r = case(ctx, claims_completeness=0.55, **scores(3.2, -0.9))
    assert r.cause == "DATA_COMPLETENESS" and "claims_completeness=0.55" in r.evidence


def test_incomplete_history_cannot_explain_a_HIGHER_ds_score(ctx):
    r = case(ctx, claims_completeness=0.55, **scores(3.2, +2.0))
    assert r.cause != "DATA_COMPLETENESS"


def test_stale_ds_snapshot_with_claims_in_gap(ctx):
    m, *_ = ctx
    snap = pd.Timestamp(C.SNAPSHOT_DATE)
    r = case(ctx, ds_snapshot_date=snap - pd.Timedelta(days=60), gap_claim_load=1.5, **scores(3.4, -1.6))
    assert r.cause == "SNAPSHOT_MISMATCH" and "DS_stale" in r.evidence


def test_stale_actuarial_snapshot_flips_the_sign(ctx):
    snap = pd.Timestamp(C.SNAPSHOT_DATE)
    r = case(ctx, act_snapshot_date=snap - pd.Timedelta(days=45), gap_claim_load=1.2, **scores(2.0, +1.3))
    assert r.cause == "SNAPSHOT_MISMATCH" and "ACT_stale" in r.evidence


def test_snapshot_gap_alone_is_not_a_cause(ctx):
    snap = pd.Timestamp(C.SNAPSHOT_DATE)
    r = case(ctx, ds_snapshot_date=snap - pd.Timedelta(days=60), gap_claim_load=0.0, **scores(3.0, -0.8))
    assert r.cause != "SNAPSHOT_MISMATCH"


def test_duplicate_inflation_is_critical_tier(ctx):
    r = case(ctx, claim_lines_total=40, claim_lines_distinct=26, **scores(2.0, +0.9))
    assert r.cause == "DUPLICATE_INPUT" and r.tier == "Critical"


def test_model_version_change_uses_the_changelog(ctx):
    r = case(ctx, ds_score_version="v2.4", rx_count=14, **scores(2.8, +0.7))
    assert r.cause == "MODEL_VERSION" and "rx_count" in r.evidence


def test_clean_diagnostics_with_huge_gap_stays_unexplained(ctx):
    r = case(ctx, **scores(3.0, -2.2))
    assert r.cause == "UNEXPLAINED" and r.unexplained_residual > 1.5


def test_confidence_orders_unexplained_above_expected(ctx):
    expected = case(ctx, **scores(3.0, +0.48))
    unexplained = case(ctx, **scores(3.0, -2.2))
    assert unexplained.confidence > expected.confidence


# ---------------------------------------------------------------- copilot guardrails
def test_deterministic_summary_passes_citation_check(ctx):
    r = case(ctx, claims_completeness=0.55, **scores(3.2, -0.9))
    packet = evidence_packet(r)
    expl = deterministic_summary(packet)
    assert validate_explanation(expl, packet) == []
    assert expl["category"] == "DATA_COMPLETENESS" and "claims_completeness=0.55" in expl["evidence"]


def test_copilot_rejects_invented_evidence(ctx):
    r = case(ctx, claims_completeness=0.55, **scores(3.2, -0.9))
    packet = evidence_packet(r)
    bad = {"category": "DATA_COMPLETENESS", "evidence": ["claims_completeness=0.30"], "next_step": "x"}
    assert validate_explanation(bad, packet)
    unknown = {"category": "DATA_COMPLETENESS", "evidence": ["provider_fraud_score=0.9"], "next_step": "x"}
    assert validate_explanation(unknown, packet)


def test_copilot_rejects_defect_with_no_citation(ctx):
    packet = evidence_packet(case(ctx, claims_completeness=0.55, **scores(3.2, -0.9)))
    assert validate_explanation({"category": "SNAPSHOT_MISMATCH", "evidence": [], "next_step": "x"}, packet)
    assert validate_explanation({"category": "UNEXPLAINED", "evidence": [], "next_step": "x"}, packet) == []
