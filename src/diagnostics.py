"""Deterministic diagnostics and root-cause classification.

For every member beyond tolerance we ask, for each known defect mechanism:
    "If this mechanism is real for this member, how much of delta would it explain?"
Each mechanism has (a) a gate -- a diagnostic that must fire -- and (b) an *expected impact*,
estimated with sensitivities fitted on clean members only (no ground truth is ever used).

    explained_fraction_k = clip(expected_delta_k / delta, 0, 1)      (sign-consistent only)

Decision rule, per flagged member
    1. If the best defect explains >= MIN_EXPLAINED of delta  -> that defect is the cause.
    2. Otherwise compare delta with clean peers of similar utilization (robust z):
         |z| <= ROBUST_Z_MAX -> EXPECTED_MODEL_VARIANCE   (ordinary methodology difference)
         |z| >  ROBUST_Z_MAX -> UNEXPLAINED                (kept visible, never absorbed)

The unexplained residual is retained as a column so it can be measured and monitored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .reconciliation import add_delta, utilization_load, impact, priority_score

AGE_ORDER = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-64"]
PLANTED_TO_LABEL = {"METHOD": "EXPECTED_MODEL_VARIANCE", "UNEXPLAINED_SHOCK": "UNEXPLAINED",
                    "DATA_COMPLETENESS": "DATA_COMPLETENESS", "SNAPSHOT_MISMATCH": "SNAPSHOT_MISMATCH",
                    "DUPLICATE_INPUT": "DUPLICATE_INPUT", "MODEL_VERSION": "MODEL_VERSION"}


# ------------------------------------------------------------------ features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = add_delta(df)
    for c in ["last_claim_date", "ds_snapshot_date", "act_snapshot_date"]:
        d[c] = pd.to_datetime(d[c])
    d["load"] = utilization_load(d)
    d["age_idx"] = d["age_band"].map({a: i for i, a in enumerate(AGE_ORDER)}).astype(float)
    # >0 : DS snapshot is OLDER than the actuarial one (DS stale); <0 : actuarial is stale
    d["snapshot_gap_days"] = (d["act_snapshot_date"] - d["ds_snapshot_date"]).dt.days
    d["dup_claim_ratio"] = (1 - d["claim_lines_distinct"] / d["claim_lines_total"].replace(0, np.nan)).fillna(0.0)
    d["version_mismatch"] = d["ds_score_version"] != C.EXPECTED_DS_VERSION
    d["expo"] = d["months_enrolled"] / 12.0
    return d


def clean_mask(d: pd.DataFrame) -> pd.Series:
    """Members with no diagnostic red flag. Used to fit sensitivities/baselines (not outcomes)."""
    return ((d["claims_completeness"] >= 0.90) & (d["dup_claim_ratio"] < 0.03)
            & (d["snapshot_gap_days"] == 0) & (~d["version_mismatch"]))


# ------------------------------------------------------------------ sensitivities & baseline
PRIOR_STRENGTH = 1.0     # empirical-Bayes weight (in "exposure units") on the peer-expected load


@dataclass
class Sensitivity:
    b_ds: float      # DS score points per unit of log1p(load)
    b_act: float     # actuarial score points per unit of load
    n_fit: int = 0
    load_coef: tuple = (0.0, 0.0, 0.0)   # peer-expected load ~ expo, expo*cond, expo*age_idx


def peer_expected_load(d: pd.DataFrame, coef) -> np.ndarray:
    return np.clip(coef[0] * d["expo"] + coef[1] * d["expo"] * d["condition_count"]
                   + coef[2] * d["expo"] * d["age_idx"], 0.0, None).to_numpy()


def fit_sensitivity(d: pd.DataFrame) -> Sensitivity:
    c = d.loc[clean_mask(d)]
    X_ds = np.column_stack([np.ones(len(c)), c["age_idx"], c["condition_count"],
                            c["condition_count"] * c["age_idx"], np.log1p(c["load"])])
    X_ac = np.column_stack([np.ones(len(c)), c["age_idx"], c["condition_count"], c["load"]])
    b_ds = np.linalg.lstsq(X_ds, c["ds_risk_score"].to_numpy(), rcond=None)[0][-1]
    b_ac = np.linalg.lstsq(X_ac, c["actuarial_risk_score"].to_numpy(), rcond=None)[0][-1]
    X_ld = np.column_stack([c["expo"], c["expo"] * c["condition_count"], c["expo"] * c["age_idx"]])
    coef = np.linalg.lstsq(X_ld, c["load"].to_numpy(), rcond=None)[0]
    return Sensitivity(float(b_ds), float(b_ac), int(len(c)), tuple(float(x) for x in coef))


@dataclass
class Baseline:
    edges: np.ndarray
    median: np.ndarray
    mad: np.ndarray

    def robust_z(self, load, delta):
        s = np.clip(np.digitize(load, self.edges[1:-1]), 0, len(self.median) - 1)
        return (np.asarray(delta) - self.median[s]) / (1.4826 * self.mad[s])


def fit_baseline(d: pd.DataFrame, n_strata: int = C.N_LOAD_STRATA) -> Baseline:
    c = d.loc[clean_mask(d)]
    edges = np.unique(np.quantile(c["load"], np.linspace(0, 1, n_strata + 1)))
    s = np.clip(np.digitize(c["load"], edges[1:-1]), 0, len(edges) - 2)
    med, mad = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        x = c.loc[s == k, "delta"]
        med[k] = x.median()
        mad[k] = max((x - x.median()).abs().median(), 0.02)
    return Baseline(edges, med, mad)


# ------------------------------------------------------------------ explained fractions
def expected_deltas(d: pd.DataFrame, sens: Sensitivity) -> pd.DataFrame:
    """Signed expected contribution (score points, in delta = DS - ACT space) of each defect,
    zero unless its diagnostic gate fires."""
    L = d["load"].to_numpy()
    out = pd.DataFrame(index=d.index)

    c = d["claims_completeness"].to_numpy()
    gate = c < C.COMPLETENESS_MIN
    # Missing load given what we observed: empirical-Bayes blend of the observed load and the
    # peer-expected load (Poisson thinning => the unseen share is (1 - c) of the true rate).
    L_pred = peer_expected_load(d, sens.load_coef)
    L_hat = (L + PRIOR_STRENGTH * L_pred) / (c + PRIOR_STRENGTH)
    missing = (1 - c) * L_hat
    out["DATA_COMPLETENESS"] = np.where(gate, -sens.b_ds * (np.log1p(L + missing) - np.log1p(L)), 0.0)

    gap = d["snapshot_gap_days"].to_numpy(); g = d["gap_claim_load"].to_numpy()
    gate = (np.abs(gap) >= C.SNAPSHOT_GAP_MIN_DAYS) & (g > 0)
    ds_stale = -sens.b_ds * (np.log1p(L + g) - np.log1p(L))        # DS missed claims -> delta negative
    act_stale = sens.b_act * g                                      # ACT missed claims -> delta positive
    out["SNAPSHOT_MISMATCH"] = np.where(gate, np.where(gap > 0, ds_stale, act_stale), 0.0)

    r = d["dup_claim_ratio"].to_numpy()
    gate = r >= C.DUP_RATIO_MIN
    out["DUPLICATE_INPUT"] = np.where(gate, sens.b_ds * (np.log1p(L) - np.log1p(L * (1 - r))), 0.0)

    vm = d["version_mismatch"].to_numpy()
    scale = d["ds_score_version"].map(lambda v: C.VERSION_CHANGELOG.get(v, {}).get("scale", 1.0)).to_numpy()
    extra = C.W_RX * d["rx_count"].to_numpy() * (scale - 1.0)       # changelog: rx_count re-defined
    out["MODEL_VERSION"] = np.where(vm, sens.b_ds * (np.log1p(L + extra) - np.log1p(L)), 0.0)
    return out


def explained_fractions(d: pd.DataFrame, sens: Sensitivity) -> pd.DataFrame:
    exp = expected_deltas(d, sens)
    delta = d["delta"].to_numpy()
    safe = np.where(np.abs(delta) > 1e-9, delta, np.nan)
    frac = exp.div(safe, axis=0).clip(lower=0, upper=1).fillna(0.0)   # negative => wrong direction => 0
    return frac


# ------------------------------------------------------------------ classification
def _evidence(row, cause: str) -> str:
    if cause == "DATA_COMPLETENESS":
        return f"claims_completeness={row.claims_completeness:.2f}; delta_sign=DS_lower"
    if cause == "SNAPSHOT_MISMATCH":
        side = "DS_stale" if row.snapshot_gap_days > 0 else "ACT_stale"
        return f"snapshot_gap_days={abs(int(row.snapshot_gap_days))}; gap_claim_load={row.gap_claim_load:.2f}; {side}"
    if cause == "DUPLICATE_INPUT":
        return f"dup_claim_ratio={row.dup_claim_ratio:.2f}; delta_sign=DS_higher"
    if cause == "MODEL_VERSION":
        return f"ds_score_version={row.ds_score_version}; changed_feature=rx_count; rx_count={int(row.rx_count)}"
    if cause == "EXPECTED_MODEL_VARIANCE":
        return f"robust_z={row.robust_z:.1f}; diagnostics_clean; within clean-peer range"
    return f"robust_z={row.robust_z:.1f}; no_diagnostic_gate_explains_delta"


def classify(df: pd.DataFrame, sens: Sensitivity | None = None, baseline: Baseline | None = None) -> pd.DataFrame:
    d = add_features(df) if "load" not in df else df.copy()
    sens = sens or fit_sensitivity(d)
    baseline = baseline or fit_baseline(d)

    frac = explained_fractions(d, sens)
    for k in C.DEFECT_CAUSES:
        d[f"explained_{k}"] = frac[k]
    d["top_defect"] = frac.idxmax(axis=1)
    d["explained_fraction"] = frac.max(axis=1)
    d["explained_total"] = frac.sum(axis=1).clip(upper=1.0)
    d["unexplained_residual"] = d["abs_delta"] * (1 - d["explained_total"])
    d["robust_z"] = baseline.robust_z(d["load"].to_numpy(), d["delta"].to_numpy())
    d["contributing_causes"] = frac.apply(lambda r: "|".join(r.index[r >= 0.15]), axis=1)

    defect = d["explained_fraction"] >= C.MIN_EXPLAINED
    expected = d["robust_z"].abs() <= C.ROBUST_Z_MAX
    cause = np.where(defect, d["top_defect"], np.where(expected, "EXPECTED_MODEL_VARIANCE", "UNEXPLAINED"))
    d["cause"] = np.where(d["beyond_tolerance"], cause, "WITHIN_TOLERANCE")

    # confidence: how strongly does the evidence say something is actually wrong?
    conf = np.where(defect, 0.5 + 0.5 * d["explained_fraction"],
                    np.where(expected, C.CONFIDENCE_EXPECTED, C.CONFIDENCE_UNEXPLAINED))
    d["confidence"] = np.where(d["beyond_tolerance"], conf, 0.0)
    d["impact"] = impact(d["ds_risk_score"], d["actuarial_risk_score"])
    d["priority"] = np.where(d["beyond_tolerance"], priority_score(d["abs_delta"], d["impact"], d["confidence"]), 0.0)

    d["evidence"] = ""
    idx = d.index[d["beyond_tolerance"]]
    d.loc[idx, "evidence"] = [_evidence(r, r.cause) for r in d.loc[idx].itertuples()]
    d["tier"] = d["cause"].map(C.TIER).fillna("")
    return d


# ------------------------------------------------------------------ evaluation against planted truth
def recovery_report(classified: pd.DataFrame, gt: pd.DataFrame):
    """Compare the framework's cause to the planted primary cause, flagged members only."""
    m = classified.loc[classified["beyond_tolerance"], ["member_id", "cause"]].merge(
        gt[["member_id", "planted_primary"]], on="member_id")
    m["planted"] = m["planted_primary"].map(PLANTED_TO_LABEL)
    conf = pd.crosstab(m["planted"], m["cause"]).reindex(index=C.CAUSES, columns=C.CAUSES, fill_value=0)
    rows = []
    for k in C.CAUSES:
        tp = conf.loc[k, k]
        rows.append({"cause": k, "planted_flagged": int(conf.loc[k].sum()), "classified": int(conf[k].sum()),
                     "recall": tp / conf.loc[k].sum() if conf.loc[k].sum() else np.nan,
                     "precision": tp / conf[k].sum() if conf[k].sum() else np.nan})
    metrics = pd.DataFrame(rows)
    acc = float((m["planted"] == m["cause"]).mean())
    return conf, metrics, acc
