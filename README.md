# RiskRecon

**A synthetic validation and reconciliation framework, inspired by the kinds of problems described publicly by Oscar Health's Clinical Risk Adjustment Data Science team.**

> **Question:** When two risk estimates disagree, how do we determine *why*, and decide what deserves investigation?

**Built with:** Python · SQL (DuckDB) · pandas · statistical validation
**Data:** 100% synthetic. No Oscar or patient data. Not a reproduction of HHS-HCC or any proprietary model.
**Why I built it:** After applying for Oscar's Data Scientist I role, I wanted to explore one of the problems the team publicly described rather than simply say I was interested in it.

📄 **[Read the 4-page case study](case_study/RiskRecon_Oscar.pdf)**

---

## The idea

Don't just calculate Δ. **Explain Δ.**

```
Member + utilization data → Data validation → DS score ↔ Actuarial score
   → Reconciliation → Root-cause classification → Prioritized investigation queue → Monitoring
```

I generate 25,000 synthetic members, score them with two toy models, and **plant six disagreement mechanisms plus one control**. Then I test whether the framework recovers the reason I planted.

| Planted mechanism | Framework should say |
|---|---|
| Missing-data discrepancy | `DATA_COMPLETENESS` |
| Recency discrepancy (older snapshot) | `SNAPSHOT_MISMATCH` |
| Population-segment discrepancy (high utilization) | `EXPECTED_MODEL_VARIANCE` |
| Transformation/version change | `MODEL_VERSION` |
| Duplicate records inflate a feature | `DUPLICATE_INPUT` |
| Legitimate model disagreement | `EXPECTED_MODEL_VARIANCE` |
| *Control:* unexplainable shocks | `UNEXPLAINED` (must stay visible) |

## Results (seed 42)

| | |
|---|---|
| Members beyond tolerance | 6.9% (1,727 / 25,000) |
| Incomplete-claims members (<0.80) vs rest | 3.9× more likely to exceed tolerance |
| Cause agreement with planted truth (flagged members) | 88% |
| Recall: snapshot / version / duplicate / planted-unexplained | 100% / 100% / 99% / 99% |
| Recall: data completeness / methodology | 77% / 72% |
| Validation gates | 10 PASS · 4 WARN · 1 FAIL |

**Read these honestly.** The framework is graded on data I generated, against a classifier I designed, so this tests internal consistency, not real-world performance. Two thresholds (`MIN_EXPLAINED=0.25`, `ROBUST_Z_MAX=4.0`) were chosen after a small sensitivity grid on this data; agreement stayed between 84% and 89% across it. The misses are visible in `outputs/recovery_confusion.csv`.

## How classification works

For each flagged member, each defect has a **diagnostic gate** and an **expected impact** estimated from sensitivities fitted on clean members only (no ground truth is ever an input). A defect is named only if it explains enough of Δ *in the right direction*. Otherwise the gap is compared with clean peers of similar utilization: ordinary variance, or `UNEXPLAINED`. The unexplained residual is stored per member and has its own gate, so it is never absorbed into an aggregate.

**Priority = |Δ| × Impact × Confidence**, squashed to 0–1. This is my proposed framework, not a description of how any team prioritizes work. Decision bands and weights are assumptions (see `src/config.py`).

## Run it

```bash
pip install -r requirements.txt
python -m src.pipeline                 # data, results, outputs/, figures
python case_study/build_case_study.py  # rebuilds the PDF from outputs/
pytest -q                              # 34 tests, incl. SQL/Python parity
duckdb < sql/01_validation.sql         # also: 02_reconciliation.sql, 03_investigation_queue.sql
```

## Layout

```
data/        synthetic_members.csv, reconciliation_results.csv, ground_truth_EVAL_ONLY.csv
sql/         01_validation · 02_reconciliation · 03_investigation_queue  (DuckDB)
src/         synthetic · validation · reconciliation · diagnostics · copilot · plots · pipeline · config
notebooks/   01_generate_data · 02_validate · 03_reconcile  (executed)
tests/       reconciliation, validation, diagnostics, SQL parity
outputs/     investigation_queue.csv, cause_table.csv, validation_gates.csv, metrics.json, ...
case_study/  RiskRecon_Oscar.pdf (+ builder and figures)
```

## Optional: Reconciliation Copilot

`src/copilot.py` lets an LLM summarize one flagged case from structured diagnostics only. It never judges which score is correct, and must cite evidence verbatim (`claims_completeness=0.61`); invented evidence is rejected and replaced by a deterministic summary. The guardrail is unit-tested. **No LLM calls were made for any result here.**

## Limitations

- Toy scoring functions; no clinical content. Effect sizes are artifacts of my generator.
- Assumes DS is built from a raw claims extract while actuarial sees completed claims.
- Drift gate compares DS vs actuarial distributions (no prior snapshot exists).
- Single seed; no confidence intervals on the headline numbers.
