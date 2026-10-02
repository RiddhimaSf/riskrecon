"""Reconciliation: quantify disagreement, test whether it is structured, and rank what to
investigate first.

    delta_i    = DS_i - Actuarial_i
    beyond_i   = |delta_i| > max(ABS_TOL, REL_TOL * mean(DS_i, Actuarial_i))
    priority_i = |delta_i| x Impact_i x Confidence_i       (squashed to [0, 1])

The priority score is a PROPOSED framework, not a description of how any real team ranks work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ core quantities
def utilization_load(df) -> pd.Series:
    return C.W_INP * df["inpatient_visits"] + C.W_ED * df["ed_visits"] + C.W_RX * df["rx_count"]


def tolerance(mean_score, abs_tol: float = C.ABS_TOL, rel_tol: float = C.REL_TOL):
    """Allowed |delta| for a pair of scores: the larger of an absolute and a relative band."""
    return np.maximum(abs_tol, rel_tol * np.asarray(mean_score, float))


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta"] = out["ds_risk_score"] - out["actuarial_risk_score"]
    out["abs_delta"] = out["delta"].abs()
    out["mean_score"] = (out["ds_risk_score"] + out["actuarial_risk_score"]) / 2
    out["tolerance"] = tolerance(out["mean_score"])
    out["beyond_tolerance"] = out["abs_delta"] > out["tolerance"]
    return out


def headline_metrics(df: pd.DataFrame) -> dict:
    d = df if "delta" in df else add_delta(df)
    return {
        "n_members": int(len(d)),
        "pct_beyond_tolerance": float(d["beyond_tolerance"].mean()),
        "n_beyond_tolerance": int(d["beyond_tolerance"].sum()),
        "median_delta": float(d["delta"].median()),
        "mae": float(d["abs_delta"].mean()),
        "p95_abs_delta": float(d["abs_delta"].quantile(0.95)),
    }


# ------------------------------------------------------------------ is disagreement structured?
def _bin_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df if "delta" in df else add_delta(df)
    ld = utilization_load(d)
    d = d.assign(
        completeness_band=pd.cut(d["claims_completeness"], [0, .7, .8, .9, 1.01],
                                 labels=["<0.70", "0.70-0.80", "0.80-0.90", ">=0.90"], right=False),
        utilization_band=pd.cut(ld, [-1, 0, .5, 1.5, np.inf],
                                labels=["none", "low", "medium", "high"]),
        enrollment_band=pd.cut(d["months_enrolled"], [0, 3, 6, 11, 12],
                               labels=["1-3 mo", "4-6 mo", "7-11 mo", "12 mo"]),
    )
    return d


def segment_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    d = _bin_frame(df)
    g = d.groupby(by, observed=True)
    t = pd.DataFrame({
        "n": g.size(),
        "mean_abs_delta": g["abs_delta"].mean(),
        "median_delta": g["delta"].median(),
        "pct_beyond_tolerance": g["beyond_tolerance"].mean(),
    })
    base = d["beyond_tolerance"].mean()
    t["lift_vs_overall"] = t["pct_beyond_tolerance"] / base
    return t


def segment_tables(df: pd.DataFrame) -> dict:
    return {b: segment_table(df, b) for b in ["completeness_band", "utilization_band", "enrollment_band"]}


def completeness_lift(df: pd.DataFrame) -> dict:
    """How much more likely are incomplete-history members to exceed tolerance?"""
    d = df if "delta" in df else add_delta(df)
    inc = d["claims_completeness"] < C.COMPLETENESS_MIN
    r_inc, r_ok = d.loc[inc, "beyond_tolerance"].mean(), d.loc[~inc, "beyond_tolerance"].mean()
    return {"rate_incomplete": float(r_inc), "rate_complete": float(r_ok),
            "lift": float(r_inc / r_ok), "n_incomplete": int(inc.sum())}


# ------------------------------------------------------------------ priority
def impact(ds, act) -> np.ndarray:
    """Would the disagreement plausibly change a downstream decision? In [0, 1].

    0.6 if the two scores land in different (hypothetical) decision bands
    0.4 scaled by the higher score (more exposure at stake for higher-risk members)
    """
    ds, act = np.asarray(ds, float), np.asarray(act, float)
    crosses = np.digitize(ds, C.SCORE_BANDS) != np.digitize(act, C.SCORE_BANDS)
    exposure = np.minimum(1.0, np.maximum(ds, act) / C.SCORE_CAP)
    return 0.6 * crosses + 0.4 * exposure


def priority_score(abs_delta, impact_, confidence) -> np.ndarray:
    """|delta| x Impact x Confidence, squashed with 1 - exp(-x/K) so the 0-1 scale is fixed
    across snapshots rather than relative to this run's worst case."""
    raw = np.asarray(abs_delta, float) * np.asarray(impact_, float) * np.asarray(confidence, float)
    return 1.0 - np.exp(-raw / C.PRIORITY_K)


QUEUE_COLUMNS = ["member_id", "ds_risk_score", "actuarial_risk_score", "delta", "cause",
                 "explained_fraction", "evidence", "impact", "confidence", "priority"]


def investigation_queue(classified: pd.DataFrame, top_n: int = C.QUEUE_SIZE) -> pd.DataFrame:
    """The data product: the top-N cases to investigate first, with the suspected cause."""
    q = classified.loc[classified["beyond_tolerance"]].sort_values("priority", ascending=False).head(top_n)
    q = q[QUEUE_COLUMNS].reset_index(drop=True)
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    return q


def cause_table(classified: pd.DataFrame) -> pd.DataFrame:
    f = classified.loc[classified["beyond_tolerance"]]
    t = f.groupby("cause").agg(flagged_members=("member_id", "size"),
                               mean_abs_delta=("abs_delta", "mean"),
                               mean_priority=("priority", "mean"),
                               total_priority=("priority", "sum")).reset_index()
    t["share_of_flagged"] = t["flagged_members"] / t["flagged_members"].sum()
    t["tier"] = t["cause"].map(C.TIER)
    order = {c: i for i, c in enumerate(["DUPLICATE_INPUT", "DATA_COMPLETENESS", "SNAPSHOT_MISMATCH",
                                         "MODEL_VERSION", "UNEXPLAINED", "EXPECTED_MODEL_VARIANCE"])}
    return t.sort_values("cause", key=lambda s: s.map(order)).reset_index(drop=True)
