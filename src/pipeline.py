"""End-to-end run:  python -m src.pipeline

generate -> validate -> reconcile -> classify -> prioritize -> evaluate -> write artifacts
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as C
from .synthetic import generate
from .validation import (run_data_gates, check_distribution_drift, check_disagreement_distribution,
                         check_segment_disagreement, check_extreme_values, reconciliation_gates, gates_to_frame)
from .reconciliation import (headline_metrics, segment_tables, completeness_lift,
                             investigation_queue, cause_table)
from .diagnostics import add_features, classify, fit_sensitivity, recovery_report

ROOT = Path(__file__).resolve().parents[1]

RESULT_COLUMNS = ["member_id", "ds_risk_score", "actuarial_risk_score", "delta", "abs_delta", "beyond_tolerance",
                  "claims_completeness", "snapshot_gap_days", "gap_claim_load", "dup_claim_ratio",
                  "ds_score_version", "version_mismatch", "rx_count", "load", "months_enrolled",
                  "robust_z", "explained_DATA_COMPLETENESS", "explained_SNAPSHOT_MISMATCH",
                  "explained_DUPLICATE_INPUT", "explained_MODEL_VERSION", "explained_fraction",
                  "unexplained_residual", "cause", "evidence", "impact", "confidence", "priority"]


def run(n: int = C.N_MEMBERS, seed: int = C.SEED, write: bool = True, make_figures: bool = True) -> dict:
    members, gt = generate(n, seed)

    # 1. data gates (on the raw table, before any modelling)
    gates = run_data_gates(members)

    # 2. reconcile + classify
    feats = add_features(members)
    sens = fit_sensitivity(feats)
    classified = classify(feats, sens)

    # 3. model-output + reconciliation gates
    segs = segment_tables(classified)
    gates += [check_distribution_drift(classified), check_disagreement_distribution(classified),
              check_segment_disagreement(segs), check_extreme_values(classified)]
    gates += reconciliation_gates(classified)
    gate_df = gates_to_frame(gates)

    # 4. data product + evaluation
    queue = investigation_queue(classified)
    causes = cause_table(classified)
    conf, rec_metrics, acc = recovery_report(classified, gt)
    metrics = {
        "headline": headline_metrics(classified),
        "completeness_lift": completeness_lift(classified),
        "sensitivity": {"b_ds": sens.b_ds, "b_act": sens.b_act, "n_fit": sens.n_fit},
        "recovery_accuracy": acc,
        "queue_size": int(len(queue)),
        "queue_cause_mix": queue["cause"].value_counts().to_dict(),
        "gates": gate_df["status"].value_counts().to_dict(),
    }

    out = dict(members=members, ground_truth=gt, classified=classified, queue=queue, causes=causes,
               segments=segs, gates=gate_df, recovery_confusion=conf, recovery_metrics=rec_metrics,
               metrics=metrics, sensitivity=sens)

    if write:
        (ROOT / "data").mkdir(exist_ok=True); (ROOT / "outputs").mkdir(exist_ok=True)
        members.to_csv(ROOT / "data" / "synthetic_members.csv", index=False)
        gt.to_csv(ROOT / "data" / "ground_truth_EVAL_ONLY.csv", index=False)
        classified[RESULT_COLUMNS].to_csv(ROOT / "data" / "reconciliation_results.csv", index=False)
        queue.to_csv(ROOT / "outputs" / "investigation_queue.csv", index=False)
        causes.to_csv(ROOT / "outputs" / "cause_table.csv", index=False)
        gate_df.to_csv(ROOT / "outputs" / "validation_gates.csv", index=False)
        conf.to_csv(ROOT / "outputs" / "recovery_confusion.csv")
        rec_metrics.to_csv(ROOT / "outputs" / "recovery_metrics.csv", index=False)
        for k, t in segs.items():
            t.to_csv(ROOT / "outputs" / f"segment_{k}.csv")
        (ROOT / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        if make_figures:
            from .plots import build_all
            build_all(classified, segs, ROOT / "case_study" / "figures")
    return out


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(json.dumps(r["metrics"], indent=2, default=float))
    print(r["causes"].round(3).to_string(index=False))
    print(r["gates"][["group", "gate", "status", "value", "limit"]].to_string(index=False))
    print(r["queue"].head(10)[["rank", "member_id", "ds_risk_score", "actuarial_risk_score", "delta", "cause", "priority"]].round(3).to_string(index=False))
