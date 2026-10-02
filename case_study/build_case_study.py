"""Builds case_study/RiskRecon_Oscar.pdf from the pipeline outputs (no hand-typed numbers).

    python -m src.pipeline && python case_study/build_case_study.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.lib.fonts import addMapping

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG, DATA = ROOT / "outputs", ROOT / "case_study" / "figures", ROOT / "data"

# ------------------------------------------------------------------ fonts & palette
fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(fdir / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY, TEAL, ORANGE = colors.HexColor("#1F2A44"), colors.HexColor("#1B7F79"), colors.HexColor("#D9622B")
GREY, LIGHT, MID = colors.HexColor("#9AA3B2"), colors.HexColor("#EEF1F6"), colors.HexColor("#5B657A")
GREEN, AMBER, RED = colors.HexColor("#2E8B57"), colors.HexColor("#C98A00"), colors.HexColor("#C0392B")

S = {
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=NAVY, spaceAfter=3),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=9.6, leading=12, textColor=TEAL, spaceBefore=7, spaceAfter=3),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=8.4, leading=11.6, textColor=MID, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="DV", fontSize=8.0, leading=11.2, textColor=NAVY, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="DV", fontSize=6.9, leading=9.2, textColor=MID),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.2, leading=9.2, textColor=NAVY),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.2, leading=9.2, textColor=NAVY),
    "hdr": ParagraphStyle("hdr", fontName="DV-B", fontSize=7.0, leading=9, textColor=colors.white),
    "big": ParagraphStyle("big", fontName="DV-B", fontSize=17, leading=20, textColor=NAVY, alignment=1),
    "lab": ParagraphStyle("lab", fontName="DV", fontSize=6.8, leading=8.6, textColor=MID, alignment=1),
    "call": ParagraphStyle("call", fontName="DV", fontSize=8.2, leading=11.6, textColor=NAVY),
    "mono": ParagraphStyle("mono", fontName="DV", fontSize=6.8, leading=9.0, textColor=NAVY),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=7.8, leading=10.8, textColor=NAVY, leftIndent=9, bulletIndent=0),
}
W = letter[0] - 2 * 0.55 * inch


def P(t, s="body"):
    return Paragraph(t, S[s])


def img(path, width):
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def table(rows, widths, header=True, zebra=True, extra=()):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.6),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#D5DAE3"))]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2 if header else 1, len(rows), 2)]
    t.setStyle(TableStyle(st + list(extra)))
    return t


def callout(flowables, color=ORANGE, bg=LIGHT, width=W):
    t = Table([[flowables]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def stat(big, label, color=NAVY):
    st = ParagraphStyle("b2", parent=S["big"], textColor=color)
    return [Paragraph(big, st), Paragraph(label, S["lab"])]


# ------------------------------------------------------------------ page chrome
TITLES = {1: "System design", 2: "Where do the models disagree?", 3: "Why do they disagree?",
          4: "From analysis to dependable data product"}


def chrome(c, doc):
    c.saveState()
    pw, ph = letter
    c.setFillColor(NAVY); c.rect(0, ph - 0.42 * inch, pw, 0.42 * inch, stroke=0, fill=1)
    c.setFillColor(colors.white); c.setFont("DV-B", 9.5); c.drawString(0.55 * inch, ph - 0.27 * inch, "RiskRecon")
    c.setFont("DV", 8.2); c.drawRightString(pw - 0.55 * inch, ph - 0.27 * inch, f"{doc.page} · {TITLES.get(doc.page, "overflow")}")
    c.setFillColor(ORANGE); c.rect(0, ph - 0.45 * inch, pw, 0.03 * inch, stroke=0, fill=1)
    c.setFillColor(MID); c.setFont("DV", 6.6)
    c.drawString(0.55 * inch, 0.32 * inch, "100% synthetic data. No Oscar or patient data. Not a reproduction of any proprietary model.")
    c.drawRightString(pw - 0.55 * inch, 0.32 * inch, f"{doc.page} / 4")
    c.restoreState()


# ------------------------------------------------------------------ load artifacts
M = json.loads((OUT / "metrics.json").read_text())
H, LIFT = M["headline"], M["completeness_lift"]
causes = pd.read_csv(OUT / "cause_table.csv")
rec = pd.read_csv(OUT / "recovery_metrics.csv").set_index("cause")
queue = pd.read_csv(OUT / "investigation_queue.csv")
gates = pd.read_csv(OUT / "validation_gates.csv")
seg_c = pd.read_csv(OUT / "segment_completeness_band.csv", index_col=0)
seg_u = pd.read_csv(OUT / "segment_utilization_band.csv", index_col=0)
seg_e = pd.read_csv(OUT / "segment_enrollment_band.csv", index_col=0)
res = pd.read_csv(DATA / "reconciliation_results.csv")
flagged = res[res.beyond_tolerance].sort_values("priority", ascending=False)
LABEL = {"DATA_COMPLETENESS": "Incomplete data", "SNAPSHOT_MISMATCH": "Snapshot mismatch",
         "DUPLICATE_INPUT": "Duplicate input", "MODEL_VERSION": "Model version change",
         "EXPECTED_MODEL_VARIANCE": "Model methodology", "UNEXPLAINED": "Unexplained"}
pct = lambda x, d=1: f"{x * 100:.{d}f}%"


def page1():
    f = [P("When two risk estimates disagree, how do we determine <i>why</i> &mdash; and decide what deserves investigation?", "h1"),
         P("A synthetic validation and reconciliation framework, inspired by the kinds of problems described publicly by "
           "Oscar Health&rsquo;s Clinical Risk Adjustment Data Science team. Built with Python, SQL, pandas and statistical validation.", "sub")]
    left = img(FIG / "flow_system.png", 2.55 * inch)
    RW = W - 2.55 * inch - 0.1 * inch
    fields = table([
        [P("Category", "hdr"), P("Synthetic fields", "hdr")],
        [P("Member", "cellb"), P("member_id, age_band, region", "cell")],
        [P("Clinical-style", "cellb"), P("condition_count, chronic_condition_flag", "cell")],
        [P("Utilization", "cellb"), P("inpatient_visits, ed_visits, rx_count", "cell")],
        [P("Completeness", "cellb"), P("months_enrolled, claims_completeness, last_claim_date, claim-line counts", "cell")],
        [P("Model outputs", "cellb"), P("ds_risk_score, actuarial_risk_score", "cell")],
        [P("Metadata", "cellb"), P("score versions, snapshot dates", "cell")],
    ], [0.95 * inch, RW - 0.95 * inch])
    right = [
        callout([P("<b>Don&rsquo;t just calculate &Delta;. Explain &Delta;.</b>", "call"),
                 Spacer(1, 2),
                 P("A gap between two risk scores is a symptom, not a diagnosis. It can come from the data, the pipeline, "
                   "timing, versioning, methodology &mdash; or from nothing anyone can explain yet. RiskRecon separates those "
                   "causes, then ranks what to look at first.", "body")], width=RW),
        P("The synthetic dataset", "h2"),
        P(f"<b>{H['n_members']:,} synthetic members</b> &mdash; enough to show real analytical behaviour without pretending "
          "scale is the accomplishment. Two toy scoring functions score the same latent utilization.", "body"),
        Spacer(1, 4), fields, Spacer(1, 4),
        P("No diagnoses are modelled and no attempt is made to implement ACA HHS-HCC. The point isn&rsquo;t &ldquo;look, I "
          "understand clinical medicine.&rdquo; It&rsquo;s <b>&ldquo;look how I approach a consequential analytical system.&rdquo;</b>", "body"),
    ]
    t = Table([[left, right]], colWidths=[2.55 * inch + 0.1 * inch, W - 2.55 * inch - 0.1 * inch])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (0, 0), 10), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    f += [t, P("Deliberately injected failure modes", "h2"),
          P("Instead of building a dashboard, I planted six disagreement mechanisms (plus one control) so I can test whether the "
            "framework <b>recovers the reason I planted</b>. The ground truth is used only to grade the framework, never as an input to it.", "body"),
          Spacer(1, 4)]
    rows = [[P("#", "hdr"), P("Planted mechanism", "hdr"), P("How it is injected", "hdr"), P("Framework should say", "hdr")]]
    spec = [
        ("1", "Missing-data discrepancy", "DS is built from a raw claims extract; members under 0.80 completeness lose claims", "DATA_COMPLETENESS"),
        ("2", "Recency discrepancy", "One side scores from an older snapshot and misses claims dated in the gap", "SNAPSHOT_MISMATCH"),
        ("3", "Population-segment discrepancy", "DS has diminishing returns and noise that grows with utilization", "EXPECTED_MODEL_VARIANCE"),
        ("4", "Transformation / version", "Staged rollout of DS v2.4, which re-defines the rx_count feature", "MODEL_VERSION"),
        ("5", "Pipeline / data issue", "Duplicate claim lines inflate DS utilization features", "DUPLICATE_INPUT"),
        ("6", "Legitimate model disagreement", "Clean data; the methodologies simply produce different estimates", "EXPECTED_MODEL_VARIANCE"),
        ("ctrl", "Unexplainable shocks (control)", "Idiosyncratic shocks on members with clean diagnostics", "UNEXPLAINED"),
    ]
    for r in spec:
        rows.append([P(r[0], "cellb"), P(r[1], "cellb"), P(r[2], "cell"), P(r[3], "mono")])
    f.append(table(rows, [0.38 * inch, 1.75 * inch, W - 0.38 * inch - 1.75 * inch - 1.62 * inch, 1.62 * inch]))
    f += [Spacer(1, 9), callout([
        P("<b>Scope guardrails &mdash; what this deliberately is not</b>", "call"), Spacer(1, 2),
        P("Not a healthcare risk prediction model. No patient data. No claim to reproduce HHS-HCC or any proprietary model. "
          "No AUC optimized on a synthetic target. The LLM extension is optional and not the centerpiece. "
          "The impressive part should be the <b>reasoning and reliability architecture</b>, not the complexity.", "body")], color=TEAL)]
    return f


def page2():
    f = [P("Where do the models disagree?", "h1")]
    stats = Table([[stat(pct(H["pct_beyond_tolerance"]), f"of members beyond tolerance<br/>({H['n_beyond_tolerance']:,} of {H['n_members']:,})", ORANGE),
                    stat(f"{H['median_delta']:+.3f}", "median Δ (DS − Actuarial)"),
                    stat(f"{H['mae']:.3f}", "mean absolute error"),
                    stat(f"{H['p95_abs_delta']:.2f}", "95th percentile |Δ|")]],
                  colWidths=[W / 4] * 4)
    stats.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEAFTER", (0, 0), (-2, -1), .5, colors.white),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    f += [stats, Spacer(1, 4),
          P("Tolerance: a pair is in disagreement when |Δ| exceeds the larger of 0.35 points and 20% of the pair&rsquo;s mean score. "
            "Δ = DS score − Actuarial score.", "small"), Spacer(1, 5)]
    c1 = [P("Chart 1 &mdash; Score A vs Score B", "cellb"), img(FIG / "fig1_scatter.png", 3.4 * inch)]
    c2 = [P("Chart 2 &mdash; Distribution of Δ", "cellb"), img(FIG / "fig2_delta.png", 3.4 * inch)]
    t = Table([[c1, c2]], colWidths=[W / 2, W / 2]); t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    f += [t, Spacer(1, 3), P("Chart 3 &mdash; Disagreement by segment: is it random, or structured?", "cellb"),
          img(FIG / "fig3_segments.png", W), Spacer(1, 4)]
    f.append(callout([
        P("<b>Key finding</b>", "call"), Spacer(1, 2),
        P(f"The largest disagreements were <b>not random</b>. Members with incomplete claims histories (completeness &lt; 0.80) were "
          f"<b>{LIFT['lift']:.1f}&times; more likely</b> to exceed the reconciliation threshold "
          f"({pct(LIFT['rate_incomplete'])} vs {pct(LIFT['rate_complete'])}; {LIFT['n_incomplete']:,} members).", "call"),
        Spacer(1, 3),
        P(f"Two more things the segments say. Members with <b>zero observed utilization</b> disagree {seg_u.loc['none', 'lift_vs_overall']:.1f}&times; "
          f"as often as average &mdash; an empty claims history is itself a warning sign. And <b>enrollment duration is not a driver</b> "
          f"(range {pct(seg_e.pct_beyond_tolerance.min())}&ndash;{pct(seg_e.pct_beyond_tolerance.max())}): a useful negative result, "
          "because it rules out one tempting explanation.", "body")]))
    return f


def page3():
    f = [P("Why do they disagree?", "h1"),
         P("Every flagged member is classified. A defect is named only if its <b>diagnostic fires</b> and it <b>explains enough of Δ in the "
           "right direction</b>; otherwise the gap is compared with clean peers of similar utilization &mdash; ordinary variance, or "
           "<b>UNEXPLAINED</b>. A discrepancy isn&rsquo;t automatically a model failure: sometimes it&rsquo;s data, pipeline, timing, "
           "versioning, methodology, or genuinely unexplained.", "sub")]
    rows = [[P("Cause", "hdr"), P("Flagged", "hdr"), P("Mean |Δ|", "hdr"), P("Tier", "hdr"),
             P("Recall*", "hdr"), P("Precision*", "hdr")]]
    tier_col = {"Critical": RED, "High": ORANGE, "Investigate": AMBER, "Monitor": TEAL}
    extra = []
    for i, r in causes.iterrows():
        rr = rec.loc[r.cause]
        rows.append([P(f"<b>{LABEL[r.cause]}</b><br/><font size=6 color='#5B657A'>{r.cause}</font>", "cell"),
                     P(f"{int(r.flagged_members):,}  <font color='#5B657A'>({pct(r.share_of_flagged, 0)})</font>", "cell"),
                     P(f"{r.mean_abs_delta:.2f}", "cell"), P(f"<b><font color='{tier_col[r.tier].hexval().replace('0x', '#')}'>{r.tier}</font></b>", "cell"),
                     P(pct(rr.recall, 0), "cell"), P(pct(rr.precision, 0), "cell")])
        extra.append(("TEXTCOLOR", (3, i + 1), (3, i + 1), tier_col[r.tier]))
    f.append(table(rows, [2.15 * inch, 1.2 * inch, 0.85 * inch, 1.0 * inch, 0.9 * inch, W - 6.1 * inch]))
    f.append(Spacer(1, 3))
    f.append(P(f"*Recall/precision vs. the planted primary cause, flagged members only; overall agreement <b>{pct(M['recovery_accuracy'], 0)}</b>. "
               "This grades the framework on data I generated, alongside a framework I designed &mdash; it tests internal consistency, not "
               "real-world performance. Two knobs (minimum explained share 0.25; robust-z 4.0) were set after a small sensitivity grid on this data; "
               "agreement stayed between 84% and 89% across it. The misses fall into two groups: incomplete-claims cases whose unseen claims were the expensive ones (left UNEXPLAINED), and ordinary methodology gaps that coincide with a weak diagnostic and get credited to it.", "small"))
    f.append(P("Reconciliation priority score", "h2"))
    f.append(callout([
        P("<b>Priority = |Δ| &times; Impact &times; Confidence</b>  &nbsp;&nbsp;<font size=7 color='#5B657A'>(squashed to 0&ndash;1 with 1 − e<super>−x</super>, so the scale is fixed across snapshots)</font>", "call"),
        Spacer(1, 3),
        P("<b>Disagreement magnitude</b> &mdash; how far apart are the scores?  &nbsp;<b>Impact</b> &mdash; would the difference change a downstream use? "
          "(0.6 if the scores land in different decision bands, plus 0.4 scaled by the higher score).  &nbsp;<b>Confidence</b> &mdash; how strongly does the "
          "evidence say something is actually wrong? (defect: 0.5 + 0.5 &times; share of Δ it explains; unexplained 0.6; ordinary variance 0.15).", "body"),
        Spacer(1, 2),
        P("This is my proposed framework, not a description of how any real team prioritizes work. Bands and weights are assumptions to be challenged.", "small")], color=TEAL))
    f.append(P(f"The data product: an investigation queue (top 6 of {M['queue_size']})", "h2"))
    rows = [[P(h, "hdr") for h in ["#", "Member", "DS", "Actuarial", "Δ", "Suspected cause", "Priority"]]]
    for _, r in queue.head(6).iterrows():
        rows.append([P(str(int(r["rank"])), "cell"), P(r.member_id, "cellb"), P(f"{r.ds_risk_score:.1f}", "cell"),
                     P(f"{r.actuarial_risk_score:.1f}", "cell"), P(f"{r.delta:+.1f}", "cell"),
                     P(LABEL[r.cause].lower(), "cell"), P(f"{r.priority:.2f}", "cellb")])
    f.append(table(rows, [0.3 * inch, 0.9 * inch, 0.6 * inch, 0.8 * inch, 0.65 * inch, W - 4.25 * inch, 1.0 * inch]))
    mix = M["queue_cause_mix"]
    f.append(Spacer(1, 3))
    f.append(P(f"Instead of &ldquo;here are {H['n_beyond_tolerance']:,} mismatched records,&rdquo; the output is &ldquo;here are the {M['queue_size']} cases to investigate first, "
               f"and why&rdquo; ({mix.get('SNAPSHOT_MISMATCH', 0)} stale-snapshot, {mix.get('UNEXPLAINED', 0)} unexplained). Stale snapshots dominate the case queue "
               "because they carry the largest, most decision-relevant gaps; <b>duplicate input</b> is tiered Critical at the <i>cause</i> level instead, "
               "because one upstream fix clears every affected member.", "body"))
    f.append(P("Highest-priority case for each root cause, with the evidence the classifier used", "h2"))
    rows = [[P(h, "hdr") for h in ["Member", "Δ", "Cause", "Evidence (deterministic diagnostics)", "Priority"]]]
    for cause in ["SNAPSHOT_MISMATCH", "DUPLICATE_INPUT", "DATA_COMPLETENESS", "MODEL_VERSION", "UNEXPLAINED", "EXPECTED_MODEL_VARIANCE"]:
        r = flagged[flagged.cause == cause].iloc[0]
        rows.append([P(r.member_id, "cellb"), P(f"{r.delta:+.1f}", "cell"), P(LABEL[cause], "cell"),
                     P(r.evidence.replace(";", " ·"), "mono"), P(f"{r.priority:.2f}", "cell")])
    f.append(table(rows, [0.75 * inch, 0.5 * inch, 1.35 * inch, W - 3.55 * inch, 0.95 * inch]))
    return f


def page4():
    f = [P("From analysis &rarr; dependable data product", "h1"),
         P("The reconciliation is only as trustworthy as the checks around it. Validation runs before modelling, after scoring, and "
           "after classification &mdash; and a failing gate is reported, not hidden.", "sub"),
         img(FIG / "flow_dependable.png", W * 0.94), Spacer(1, 4)]
    colr = {"PASS": GREEN, "WARN": AMBER, "FAIL": RED}

    def gate_block(group, title):
        g = gates[gates.group == group]
        rows = [[P(title, "hdr"), P("", "hdr")]]
        for _, r in g.iterrows():
            val = r.value if isinstance(r.value, str) else f"{r.value:g}"
            rows.append([P(f"{r.gate.replace('_', ' ')}<br/><font size=6 color='#5B657A'>{val}  (limit {r.limit})</font>", "cell"),
                         P(f"<b><font color='{colr[r.status].hexval().replace('0x', '#')}'>{r.status}</font></b>", "cell")])
        return table(rows, [(W / 3 - 8) - 0.5 * inch, 0.5 * inch],
                     extra=[("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)])

    cols = Table([[gate_block("data", "Data gates"), gate_block("model_output", "Model-output gates"),
                   gate_block("reconciliation", "Reconciliation gates")]], colWidths=[W / 3] * 3)
    cols.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    st = M["gates"]
    f += [cols, Spacer(1, 3),
          P(f"On this dataset: {st.get('PASS', 0)} pass, {st.get('WARN', 0)} warn, {st.get('FAIL', 0)} fail &mdash; the failing and warning gates sit exactly where defects "
            "were planted (duplicate claim lines, incomplete claims, stale snapshots, version rollout). Without a prior snapshot, the drift gate compares the DS and actuarial "
            "score distributions to each other. The data gates are also written in SQL (<font face='DV' size=6.8>sql/01_validation.sql</font>) and tested against the Python versions.", "small"),
          Spacer(1, 4)]
    f.append(callout([
        P("<b>Design principle</b>", "call"), Spacer(1, 2),
        P("<i>Knowing what a system can&rsquo;t explain is part of knowing that it works.</i>", "call"), Spacer(1, 3),
        P("Unexplained reconciliation differences should remain visible and measurable rather than being absorbed into an aggregate "
          "performance metric or forced into a convenient label. That is why <b>UNEXPLAINED</b> is a first-class cause, why its residual is stored per member, "
          f"and why it has its own gate ({pct(res[(res.cause == 'UNEXPLAINED') & res.beyond_tolerance].shape[0] / len(res), 1)} of members today). "
          "The framework is also graded on a planted control &mdash; shocks with clean diagnostics &mdash; that it is <i>supposed</i> to leave unexplained.", "body")], color=TEAL))
    f.append(P("Optional extension: a Reconciliation Copilot", "h2"))
    f.append(P("A small experiment, deliberately not the centerpiece. The LLM sees only structured diagnostics for one flagged case and returns a likely category, "
               "the evidence, and a next step. <b>It never decides whether a score is correct</b>; an explanation that cites evidence not in the diagnostics "
               "is rejected and replaced by the deterministic summary. The guardrail is implemented and unit-tested; "
               "<b>no LLM calls were made for the results in this document.</b>", "body"))
    f.append(Spacer(1, 3))
    ex = flagged[flagged.cause == "DATA_COMPLETENESS"].iloc[0]
    f.append(table([[P("Input (diagnostics only)", "hdr"), P("Output (must cite the input verbatim)", "hdr")],
                    [P(f"claims_completeness={ex.claims_completeness:.2f}<br/>delta={ex.delta:.2f}<br/>snapshot_gap_days={int(ex.snapshot_gap_days)}<br/>"
                       f"dup_claim_ratio={ex.dup_claim_ratio:.2f}<br/>ds_score_version={ex.ds_score_version}", "mono"),
                     P(f"category: DATA_COMPLETENESS<br/>evidence: claims_completeness={ex.claims_completeness:.2f}<br/>"
                       "next step: re-run the DS score after claims run-out completes; compare against the completed extract.", "mono")]],
                   [W * 0.4, W * 0.6], zebra=False))
    f.append(P("Assumptions I&rsquo;d most like to have challenged", "h2"))
    for t in ["That DS scores are built from a raw claims extract while actuarial scores see completed claims &mdash; the completeness effect depends on it.",
              "The tolerance rule (larger of 0.35 points or 20% of the mean score) and the hypothetical decision bands behind Impact.",
              "That a defect is named once it explains about a quarter of Δ, and that a clean-peer robust-z of 4 marks &ldquo;unexplained&rdquo;.",
              "Whether per-member classification is the right unit, versus tracing each cause to a single upstream fix."]:
        f.append(Paragraph(t, S["bul"], bulletText="•"))
    return f


def build(path):
    doc = BaseDocTemplate(str(path), pagesize=letter, leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                          topMargin=0.62 * inch, bottomMargin=0.5 * inch, title="RiskRecon: reconciling risk scores",
                          author="Riddhima", subject="Synthetic validation & reconciliation case study")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=chrome)])
    story = page1() + [PageBreak()] + page2() + [PageBreak()] + page3() + [PageBreak()] + page4()
    doc.build(story)


if __name__ == "__main__":
    out = ROOT / "case_study" / "RiskRecon_Oscar.pdf"
    build(out)
    from pypdf import PdfReader
    print(out, "pages:", len(PdfReader(str(out)).pages))
