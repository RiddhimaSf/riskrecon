"""Synthetic member generator with deliberately planted disagreement mechanisms.

Nothing here is clinical. Two toy scoring functions ("Data Science" and "Actuarial") score
the same latent utilization; six mechanisms (plus one control) are then injected so we can
test whether the reconciliation framework recovers the reason we planted.

Returns
-------
members      : what an analyst would actually see (no ground truth).
ground_truth : per-member contribution of each planted mechanism to delta = DS - Actuarial.
               Used ONLY to evaluate the framework, never as an input to it.

Planted mechanisms
------------------
1. DATA_COMPLETENESS  DS is built from a raw claims extract; incomplete history => lower DS score.
2. SNAPSHOT_MISMATCH  One side scores from an older snapshot and misses claims in the gap.
3. SEGMENT (METHOD)   DS has diminishing returns + heteroscedastic noise => bigger disagreement
                      for high-utilization members (a legitimate, structured difference).
4. MODEL_VERSION      A staged rollout re-defines rx_count in DS v2.4.
5. DUPLICATE_INPUT    Duplicate claim lines inflate DS utilization features.
6. METHOD             Clean data, different methodology => ordinary disagreement.
7. UNEXPLAINED_SHOCK  (control) Idiosyncratic shocks with clean diagnostics. A good framework
                      should NOT be able to explain these; it should leave them visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

AGE_BANDS = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-64"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
MECHANISMS = ["METHOD", "DATA_COMPLETENESS", "SNAPSHOT_MISMATCH",
              "DUPLICATE_INPUT", "MODEL_VERSION", "UNEXPLAINED_SHOCK"]


def load(inp, ed, rx):
    """Utilization 'load' (cost-proxy weights from config; not clinical)."""
    return C.W_INP * inp + C.W_ED * ed + C.W_RX * rx


def _ds_score(age_idx, cond, ld):
    """DS toy model: age x condition interaction + diminishing returns on utilization."""
    return 0.40 + 0.17 * age_idx + 0.42 * cond + 0.04 * cond * age_idx + 1.9 * np.log1p(ld)


def _act_score(age_idx, cond, ld):
    """Actuarial toy model: additive and linear."""
    return 0.55 + 0.20 * age_idx + 0.50 * cond + 1.0 * ld


def _days(x):
    return np.asarray(x).astype("timedelta64[D]")


def generate(n: int = C.N_MEMBERS, seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    snap = np.datetime64(C.SNAPSHOT_DATE)

    # ---------------- members & true utilization ----------------
    age_idx = rng.choice(6, n, p=[.10, .14, .16, .18, .22, .20])
    region = rng.choice(REGIONS, n, p=[.20, .25, .20, .15, .20])
    months = np.where(rng.random(n) < 0.72, 12, rng.integers(1, 12, n))
    expo = months / 12
    frailty = rng.gamma(1.5, 1 / 1.5, n)
    cond = rng.poisson(0.10 + 0.12 * age_idx + 0.35 * frailty)
    inp_t = rng.poisson(expo * (0.04 + 0.12 * frailty))
    ed_t = rng.poisson(expo * (0.12 + 0.26 * frailty))
    rx_t = rng.poisson(expo * (1.5 + 2.2 * cond + 1.2 * frailty))

    # ---------------- mechanism assignment ----------------
    # 1. completeness: most members ~0.95; low tail more likely for short enrollment
    low = rng.random(n) < (0.06 + 0.10 * (months <= 3))
    comp = np.where(low, np.clip(rng.beta(6, 4, n), 0.30, 0.79),
                    np.clip(rng.beta(30, 1.5, n), 0.80, 1.0))

    # 2. snapshots: real mismatch (claims in gap), benign gap (>=30d, no claims in gap), tiny gap
    u = rng.random(n)
    snap_real = u < 0.030
    snap_benign = (u >= 0.030) & (u < 0.070)
    snap_tiny = (u >= 0.070) & (u < 0.170)
    big = snap_real | snap_benign
    gap_days = np.zeros(n, int)
    gap_days[big] = rng.integers(30, 91, big.sum())
    gap_days[snap_tiny] = rng.integers(1, 8, snap_tiny.sum())
    ds_stale = rng.random(n) < 0.70
    stale_snap = snap - _days(gap_days)

    rec_inp = np.where(snap_real & (rng.random(n) < 0.40), 1, 0)
    rec_ed = np.where(snap_real, rng.poisson(0.8, n), 0)
    rec_ed = np.where(snap_real & (rec_inp + rec_ed == 0), 1, rec_ed)
    rec_rx = np.where(snap_real, rng.poisson(2.0, n), 0)
    inp_t = inp_t + rec_inp
    ed_t = ed_t + rec_ed
    rx_t = rx_t + rec_rx                                   # true totals include recent events
    gap_load = load(rec_inp, rec_ed, rec_rx)

    # 5. duplicates
    dup = rng.random(n) < 0.016
    dup_r = np.where(dup, rng.uniform(0.20, 0.70, n), 0.0)

    # 4. staged rollout of DS v2.4 (rx_count re-definition), mostly in one region + small pilot
    rollout = ((region == "Southwest") & (rng.random(n) < 0.35)) | (rng.random(n) < 0.008)
    ds_ver = np.where(rollout, "v2.4", C.EXPECTED_DS_VERSION)
    rx_scale = np.where(rollout, C.VERSION_CHANGELOG["v2.4"]["scale"], 1.0)

    # 7. control: unexplainable shocks on otherwise clean members
    shock_pool = ~(snap_real | dup | rollout | low)
    shock = shock_pool & (rng.random(n) < 0.0055)
    shock_mag = np.where(shock, rng.uniform(1.0, 2.5, n) * rng.choice([-1, 1], n), 0.0)
    shock_on_ds = rng.random(n) < 0.5

    # ---------------- scoring pipelines ----------------
    ld_true = load(inp_t, ed_t, rx_t)
    A_true = _act_score(age_idx, cond, ld_true)
    D_true = _ds_score(age_idx, cond, ld_true)
    off = (A_true - D_true).mean()                         # align toy scales on average
    D_true = D_true + off

    # stage 1: snapshot (the stale side misses recent events)
    ds_miss = snap_real & ds_stale
    s_inp, s_ed, s_rx = (np.where(ds_miss, inp_t - rec_inp, inp_t),
                         np.where(ds_miss, ed_t - rec_ed, ed_t),
                         np.where(ds_miss, rx_t - rec_rx, rx_t))
    D_snap = _ds_score(age_idx, cond, load(s_inp, s_ed, s_rx)) + off
    act_miss = snap_real & ~ds_stale
    A_snap = np.where(act_miss, _act_score(age_idx, cond, load(inp_t - rec_inp, ed_t - rec_ed, rx_t - rec_rx)), A_true)

    # stage 2: claims completeness (DS built from the raw claims extract -> thinning)
    # Members at/above the completeness gate (>= 0.80) are treated as fully ingested; below it,
    # each claim is independently missing with probability (1 - completeness).
    keep = np.where(comp < C.COMPLETENESS_MIN, comp, 1.0)
    c_inp, c_ed, c_rx = rng.binomial(s_inp, keep), rng.binomial(s_ed, keep), rng.binomial(s_rx, keep)
    D_comp = _ds_score(age_idx, cond, load(c_inp, c_ed, c_rx)) + off

    # stage 3: duplicate records inflate DS inputs
    d_inp = c_inp + rng.binomial(c_inp, dup_r)
    d_ed = c_ed + rng.binomial(c_ed, dup_r)
    d_rx = c_rx + rng.binomial(c_rx, dup_r)
    D_dup = _ds_score(age_idx, cond, load(d_inp, d_ed, d_rx)) + off

    # stage 4: transformation/version change (rx feature re-defined upstream)
    D_ver = _ds_score(age_idx, cond, load(d_inp, d_ed, d_rx * rx_scale)) + off

    # methodology noise: heteroscedastic for DS -> more disagreement at high utilization
    eps_d = rng.normal(0, 0.07 + 0.06 * ld_true)
    eps_a = rng.normal(0, 0.07, n)

    ds = np.clip(D_ver + eps_d + np.where(shock_on_ds, shock_mag, 0.0), 0.05, 15)
    act = np.clip(A_snap + eps_a - np.where(~shock_on_ds, shock_mag, 0.0), 0.05, 15)

    # ---------------- ground truth: contribution of each mechanism to delta = DS - ACT ----------------
    member_id = np.array([f"M{i + 1:05d}" for i in range(n)])
    gt = pd.DataFrame({
        "member_id": member_id,
        "c_METHOD": (D_true + eps_d) - (A_true + eps_a),
        "c_SNAPSHOT_MISMATCH": (D_snap - D_true) + (A_true - A_snap),
        "c_DATA_COMPLETENESS": D_comp - D_snap,
        "c_DUPLICATE_INPUT": D_dup - D_comp,
        "c_MODEL_VERSION": D_ver - D_dup,
        "c_UNEXPLAINED_SHOCK": shock_mag,
    })
    contrib = gt[[f"c_{m}" for m in MECHANISMS]].abs()
    gt["planted_primary"] = contrib.idxmax(axis=1).str.replace("c_", "", regex=False)
    gt["n_active_mechanisms"] = (contrib.drop(columns="c_METHOD") > 0.15).sum(axis=1)

    # ---------------- observable member table ----------------
    has_gap = snap_real | snap_benign | snap_tiny
    ds_snap_date = np.where(has_gap & ds_stale, stale_snap, snap)
    act_snap_date = np.where(has_gap & ~ds_stale, stale_snap, snap)

    last_claim = snap - _days(rng.integers(0, 300, n))
    in_gap = snap - _days((rng.random(n) * gap_days * 0.9).astype(int))
    before_gap = stale_snap - _days(rng.integers(1, 120, n))
    last_claim = np.where(snap_real, in_gap, np.where(snap_benign, before_gap, last_claim))
    last_claim = np.where((inp_t + ed_t + rx_t) == 0, np.datetime64("NaT"), last_claim)

    other = rng.poisson(expo * (4 + 2 * cond))
    distinct = c_inp + c_ed + c_rx + other
    dup_lines = ((d_inp - c_inp) + (d_ed - c_ed) + (d_rx - c_rx)
                 + rng.binomial(other, dup_r) + rng.binomial(distinct, 0.003))  # + benign repeats

    members = pd.DataFrame({
        "member_id": member_id,
        "age_band": [AGE_BANDS[i] for i in age_idx],
        "region": region,
        "condition_count": cond,
        "chronic_condition_flag": (cond >= 2).astype(int),
        "inpatient_visits": d_inp, "ed_visits": d_ed, "rx_count": d_rx,
        "months_enrolled": months,
        "claims_completeness": comp.round(3),
        "last_claim_date": pd.Series(pd.to_datetime(last_claim)).dt.date,
        "claim_lines_total": distinct + dup_lines, "claim_lines_distinct": distinct,
        "gap_claim_load": gap_load.round(3),
        "ds_risk_score": ds.round(3), "actuarial_risk_score": act.round(3),
        "ds_score_version": ds_ver, "act_score_version": C.ACT_VERSION,
        "ds_snapshot_date": pd.Series(pd.to_datetime(ds_snap_date)).dt.date,
        "act_snapshot_date": pd.Series(pd.to_datetime(act_snap_date)).dt.date,
    })
    return members, gt
