"""Validation gates: data gates, model-output gates, reconciliation gates.

Each gate returns a GateResult (PASS / WARN / FAIL). A failing gate does not stop the
pipeline -- it is surfaced in the report so downstream users know what they are trusting.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as C

REQUIRED_COLUMNS = {
    "member_id": "object", "age_band": "object", "region": "object",
    "condition_count": "int", "chronic_condition_flag": "int",
    "inpatient_visits": "int", "ed_visits": "int", "rx_count": "int",
    "months_enrolled": "int", "claims_completeness": "float",
    "last_claim_date": "date", "claim_lines_total": "int", "claim_lines_distinct": "int",
    "gap_claim_load": "float", "ds_risk_score": "float", "actuarial_risk_score": "float",
    "ds_score_version": "object", "act_score_version": "object",
    "ds_snapshot_date": "date", "act_snapshot_date": "date",
}


@dataclass
class GateResult:
    group: str
    gate: str
    status: str          # PASS | WARN | FAIL
    value: float | str
    limit: str
    detail: str = ""

    def as_dict(self):
        return asdict(self)


def _status(ok: bool, hard: bool = True) -> str:
    return "PASS" if ok else ("FAIL" if hard else "WARN")


# ------------------------------------------------------------------ data gates
def check_unique_ids(df) -> GateResult:
    n_dup = int(df["member_id"].duplicated().sum())
    return GateResult("data", "unique_member_ids", _status(n_dup == 0), n_dup, "== 0",
                      f"{n_dup} duplicated member_id rows")


def check_schema(df) -> GateResult:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    bad = []
    for c, kind in REQUIRED_COLUMNS.items():
        if c in df.columns and kind in ("int", "float"):
            if not pd.api.types.is_numeric_dtype(df[c]):
                bad.append(c)
    ok = not missing and not bad
    return GateResult("data", "expected_schema", _status(ok), len(missing) + len(bad), "== 0",
                      f"missing={missing} non_numeric={bad}" if not ok else "all required columns present")


def check_missingness(df) -> GateResult:
    cols = [c for c in REQUIRED_COLUMNS if c in df.columns and c != "last_claim_date"]
    rate = float(df[cols].isna().any(axis=1).mean()) if cols else 1.0
    lim = C.GATES["max_missing_rate"]
    return GateResult("data", "acceptable_missingness", _status(rate <= lim), round(rate, 4), f"<= {lim}",
                      "rows with any null in required fields (last_claim_date may be null: no claims)")


def check_completeness(df) -> GateResult:
    rate = float((df["claims_completeness"] < C.COMPLETENESS_MIN).mean())
    lim = C.GATES["max_low_completeness_rate"]
    return GateResult("data", "claims_completeness", _status(rate <= lim, hard=False), round(rate, 4),
                      f"<= {lim}", f"share of members with claims_completeness < {C.COMPLETENESS_MIN}")


def check_freshness(df) -> GateResult:
    snap = pd.Timestamp(C.SNAPSHOT_DATE)
    lc = pd.to_datetime(df["last_claim_date"])
    ds = pd.to_datetime(df["ds_snapshot_date"]); ac = pd.to_datetime(df["act_snapshot_date"])
    future = float(((lc > snap) | (ds > snap) | (ac > snap)).mean())
    lim = C.GATES["max_future_claim_rate"]
    return GateResult("data", "date_freshness", _status(future <= lim), round(future, 4), f"<= {lim}",
                      f"dates after the reference snapshot {C.SNAPSHOT_DATE}")


def check_duplicates(df) -> GateResult:
    ratio = 1 - df["claim_lines_distinct"] / df["claim_lines_total"].replace(0, np.nan)
    rate = float((ratio.fillna(0) >= C.DUP_RATIO_MIN).mean())
    lim = C.GATES["max_dup_member_rate"]
    return GateResult("data", "duplicate_detection", _status(rate <= lim), round(rate, 4), f"<= {lim}",
                      f"members with >= {C.DUP_RATIO_MIN:.0%} duplicate claim lines")


def check_score_ranges(df) -> GateResult:
    lo, hi = C.GATES["score_min"], C.GATES["score_max"]
    bad = float((~df["ds_risk_score"].between(lo, hi) | ~df["actuarial_risk_score"].between(lo, hi)).mean())
    return GateResult("data", "score_ranges", _status(bad == 0), round(bad, 4), "== 0",
                      f"scores outside [{lo}, {hi}] or null")


# ------------------------------------------------------------------ model-output gates
def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between two score samples (reference-quantile bins)."""
    ref = np.asarray(reference, float); cur = np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def check_distribution_drift(df, baseline: pd.DataFrame | None = None) -> GateResult:
    """If a prior-snapshot baseline is supplied, test DS score drift against it; otherwise
    compare DS and actuarial score distributions to each other (cross-model stability)."""
    if baseline is not None:
        val, what = psi(baseline["ds_risk_score"], df["ds_risk_score"]), "DS score vs prior snapshot"
    else:
        val, what = psi(df["actuarial_risk_score"], df["ds_risk_score"]), "DS vs actuarial distribution (no prior snapshot supplied)"
    lim = C.GATES["max_psi"]
    return GateResult("model_output", "score_distribution_drift", _status(val <= lim, hard=False),
                      round(val, 4), f"PSI <= {lim}", what)


def check_disagreement_distribution(df) -> GateResult:
    p95 = float((df["ds_risk_score"] - df["actuarial_risk_score"]).abs().quantile(0.95))
    lim = C.GATES["max_p95_abs_delta"]
    return GateResult("model_output", "disagreement_distribution", _status(p95 <= lim, hard=False),
                      round(p95, 3), f"p95|delta| <= {lim}", "95th percentile absolute disagreement")


def check_segment_disagreement(seg_tables: dict) -> GateResult:
    worst, where = 0.0, ""
    for name, t in seg_tables.items():
        i = t["pct_beyond_tolerance"].idxmax()
        if t.loc[i, "pct_beyond_tolerance"] > worst:
            worst, where = float(t.loc[i, "pct_beyond_tolerance"]), f"{name}={i}"
    lim = C.GATES["max_segment_beyond_rate"]
    return GateResult("model_output", "segment_disagreement", _status(worst <= lim, hard=False),
                      round(worst, 3), f"<= {lim}", f"worst segment: {where}")


def check_extreme_values(df) -> GateResult:
    hi = C.GATES["score_max"] - 1.0
    rate = float(((df["ds_risk_score"] > hi) | (df["actuarial_risk_score"] > hi)).mean())
    lim = C.GATES["max_extreme_score_rate"]
    return GateResult("model_output", "extreme_value_monitoring", _status(rate <= lim, hard=False),
                      round(rate, 4), f"<= {lim}", f"scores within 1 point of the cap ({C.GATES['score_max']})")


# ------------------------------------------------------------------ reconciliation gates
def reconciliation_gates(res: pd.DataFrame) -> list[GateResult]:
    n = len(res)
    beyond = float(res["beyond_tolerance"].mean())
    unexpl = float(((res["cause"] == "UNEXPLAINED") & res["beyond_tolerance"]).sum() / n)
    stale = float((res["snapshot_gap_days"].abs() >= C.SNAPSHOT_GAP_MIN_DAYS).mean())
    vers = float(res["version_mismatch"].mean())
    G = C.GATES
    return [
        GateResult("reconciliation", "pct_beyond_tolerance", _status(beyond <= G["max_beyond_tol_rate"], False),
                   round(beyond, 4), f"<= {G['max_beyond_tol_rate']}", "share of members beyond tolerance"),
        GateResult("reconciliation", "unexplained_disagreement_rate", _status(unexpl <= G["max_unexplained_rate"], False),
                   round(unexpl, 4), f"<= {G['max_unexplained_rate']}", "UNEXPLAINED flagged members / all members"),
        GateResult("reconciliation", "stale_snapshot_rate", _status(stale <= G["max_stale_snapshot_rate"], False),
                   round(stale, 4), f"<= {G['max_stale_snapshot_rate']}", f"snapshot gap >= {C.SNAPSHOT_GAP_MIN_DAYS} days"),
        GateResult("reconciliation", "version_mismatch_rate", _status(vers <= G["max_version_mismatch_rate"], False),
                   round(vers, 4), f"<= {G['max_version_mismatch_rate']}", f"DS version != {C.EXPECTED_DS_VERSION}"),
    ]


def run_data_gates(df) -> list[GateResult]:
    return [check_unique_ids(df), check_schema(df), check_missingness(df),
            check_completeness(df), check_freshness(df), check_duplicates(df), check_score_ranges(df)]


def gates_to_frame(results: list[GateResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in results])
