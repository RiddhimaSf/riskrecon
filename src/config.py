"""Central configuration. Every threshold in the framework lives here so it can be
reviewed, challenged and changed in one place."""

SEED = 42
N_MEMBERS = 25_000
SNAPSHOT_DATE = "2025-12-31"          # reference ("fresh") snapshot date

# --- Synthetic utilization -> "load" weights (cost-proxy weights, NOT clinical) ---
W_INP, W_ED, W_RX = 1.2, 0.5, 0.08

# --- Reconciliation tolerance: a pair is "in disagreement" when BOTH conditions hold ---
ABS_TOL = 0.35        # absolute score points
REL_TOL = 0.20        # fraction of the mean of the two scores

# --- Hypothetical downstream decision bands (assumption used only for the Impact term) ---
SCORE_BANDS = [2.0, 4.0, 6.0]
SCORE_CAP = 6.0

# --- Versioning ---
EXPECTED_DS_VERSION = "v2.3"
ACT_VERSION = "a2025.1"
# What the (synthetic) release notes say changed. Diagnostics may use this because a real
# team would have the changelog; it is not inferred from the scores.
VERSION_CHANGELOG = {"v2.4": {"feature": "rx_count", "scale": 1.8}}

# --- Diagnostic gates ---
COMPLETENESS_MIN = 0.80
DUP_RATIO_MIN = 0.08
SNAPSHOT_GAP_MIN_DAYS = 30
MIN_EXPLAINED = 0.25      # a defect must explain >= this share of |delta| to be named as the cause
ROBUST_Z_MAX = 4.0        # beyond this (vs. clean peers) with no explanation => UNEXPLAINED
N_LOAD_STRATA = 10

# --- Priority = |delta| * Impact * Confidence, squashed to [0,1] ---
PRIORITY_K = 1.0
CONFIDENCE_EXPECTED = 0.15
CONFIDENCE_UNEXPLAINED = 0.60
QUEUE_SIZE = 50

CAUSES = ["DATA_COMPLETENESS", "SNAPSHOT_MISMATCH", "DUPLICATE_INPUT",
          "MODEL_VERSION", "EXPECTED_MODEL_VARIANCE", "UNEXPLAINED"]
DEFECT_CAUSES = ["DATA_COMPLETENESS", "SNAPSHOT_MISMATCH", "DUPLICATE_INPUT", "MODEL_VERSION"]

TIER = {
    "DUPLICATE_INPUT": "Critical",
    "DATA_COMPLETENESS": "High",
    "SNAPSHOT_MISMATCH": "High",
    "MODEL_VERSION": "High",
    "UNEXPLAINED": "Investigate",
    "EXPECTED_MODEL_VARIANCE": "Monitor",
}

# --- Validation gate limits (data + model-output + reconciliation) ---
GATES = {
    "max_missing_rate": 0.01,
    "max_low_completeness_rate": 0.05,
    "max_dup_member_rate": 0.005,
    "max_future_claim_rate": 0.0,
    "score_min": 0.0, "score_max": 15.0,
    "max_psi": 0.10,
    "max_p95_abs_delta": 1.0,
    "max_segment_beyond_rate": 0.25,
    "max_extreme_score_rate": 0.001,
    "max_beyond_tol_rate": 0.10,
    "max_unexplained_rate": 0.02,       # share of ALL members
    "max_stale_snapshot_rate": 0.02,
    "max_version_mismatch_rate": 0.02,
}
