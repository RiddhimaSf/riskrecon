"""Reconciliation Copilot (OPTIONAL extension).

The LLM never decides whether a score is correct. It only summarizes deterministic
diagnostics that the pipeline has already computed, and it must cite them verbatim:

    claims_completeness=0.61
    snapshot_gap_days=47

An explanation that cites evidence not present in the packet is rejected ("citation or refuse").
A deterministic, no-LLM summary is provided as the baseline and as the fallback.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from . import config as C

ALLOWED_CATEGORIES = C.CAUSES
PACKET_FIELDS = ["member_id", "ds_risk_score", "actuarial_risk_score", "delta", "claims_completeness",
                 "snapshot_gap_days", "gap_claim_load", "dup_claim_ratio", "ds_score_version",
                 "rx_count", "robust_z", "explained_fraction", "unexplained_residual", "cause"]

NEXT_STEP = {
    "DATA_COMPLETENESS": "Re-run the DS score after claims run-out completes; compare against the completed extract.",
    "SNAPSHOT_MISMATCH": "Re-score the stale side on the fresher snapshot and confirm the delta closes.",
    "DUPLICATE_INPUT": "Deduplicate claim lines upstream and re-score; trace the ingestion job that produced the repeats.",
    "MODEL_VERSION": "Re-score on the expected version, or on both versions, to isolate the changed feature.",
    "EXPECTED_MODEL_VARIANCE": "No data defect found. Monitor; do not investigate individually.",
    "UNEXPLAINED": "Escalate for manual review. Diagnostics do not explain this gap; keep it visible in monitoring.",
}

SYSTEM_PROMPT = """You are a reconciliation assistant. You are given structured diagnostics for ONE member whose
Data Science and Actuarial risk scores disagree. You must NOT judge which score is correct.
Return JSON with keys: category, evidence, next_step.
- category: one of {cats}.
- evidence: a list of strings, each EXACTLY a key=value pair copied from the diagnostics (e.g. "claims_completeness=0.61").
- next_step: one sentence.
If the diagnostics do not support a category, return category "UNEXPLAINED" with evidence [] rather than guessing.""".format(
    cats=", ".join(ALLOWED_CATEGORIES))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def evidence_packet(row: pd.Series) -> dict:
    """Structured, deterministic diagnostics for one flagged member."""
    return {k: (_fmt(row[k]) if k not in ("member_id", "ds_score_version", "cause") else str(row[k]))
            for k in PACKET_FIELDS if k in row.index}


def build_prompt(packet: dict) -> str:
    lines = [f"{k}={v}" for k, v in packet.items() if k != "cause"]
    return "Diagnostics:\n" + "\n".join(lines)


def deterministic_summary(packet: dict) -> dict:
    """No-LLM baseline: category is the pipeline's classification; evidence is the matching diagnostics."""
    cause = packet.get("cause", "UNEXPLAINED")
    keys = {
        "DATA_COMPLETENESS": ["claims_completeness"],
        "SNAPSHOT_MISMATCH": ["snapshot_gap_days", "gap_claim_load"],
        "DUPLICATE_INPUT": ["dup_claim_ratio"],
        "MODEL_VERSION": ["ds_score_version", "rx_count"],
        "EXPECTED_MODEL_VARIANCE": ["robust_z"],
        "UNEXPLAINED": ["robust_z", "unexplained_residual"],
    }[cause]
    return {"category": cause,
            "evidence": [f"{k}={packet[k]}" for k in keys if k in packet],
            "next_step": NEXT_STEP[cause]}


def validate_explanation(expl: dict, packet: dict) -> list[str]:
    """Return a list of problems; an empty list means the explanation is acceptable."""
    problems = []
    if expl.get("category") not in ALLOWED_CATEGORIES:
        problems.append(f"category {expl.get('category')!r} not allowed")
    cited = expl.get("evidence", [])
    if expl.get("category") not in ("UNEXPLAINED", "EXPECTED_MODEL_VARIANCE") and not cited:
        problems.append("a defect category must cite at least one diagnostic")
    for item in cited:
        m = re.fullmatch(r"([a-z_]+)=(.+)", str(item))
        if not m or m.group(1) not in packet or str(packet[m.group(1)]) != m.group(2):
            problems.append(f"evidence {item!r} does not match the diagnostics")
    if not expl.get("next_step"):
        problems.append("missing next_step")
    return problems


def llm_explanation(packet: dict, client=None, model: str | None = None) -> dict:
    """Ask an LLM to summarize the diagnostics. Falls back to the deterministic summary if the
    response violates the citation rule. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    model = model or os.environ.get("RISKRECON_MODEL", "claude-sonnet-5-5")
    msg = client.messages.create(model=model, max_tokens=400, system=SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": build_prompt(packet)}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    try:
        expl = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        return {**deterministic_summary(packet), "source": "fallback:unparseable"}
    problems = validate_explanation(expl, packet)
    if problems:
        return {**deterministic_summary(packet), "source": "fallback:" + "; ".join(problems)}
    return {**expl, "source": "llm"}
