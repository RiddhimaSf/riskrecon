"""Figures for the case study: three analysis charts and two flow diagrams."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

NAVY, TEAL, ORANGE, GREY, LIGHT = "#1F2A44", "#1B7F79", "#D9622B", "#9AA3B2", "#EEF1F6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GREY,
                     "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "axes.spines.top": False, "axes.spines.right": False})


def scatter_scores(d, path):
    fig, ax = plt.subplots(figsize=(3.55, 3.4), dpi=220)
    ok, bad = d[~d.beyond_tolerance], d[d.beyond_tolerance]
    ax.scatter(ok.ds_risk_score, ok.actuarial_risk_score, s=1.2, c=GREY, alpha=.35, lw=0, rasterized=True)
    ax.scatter(bad.ds_risk_score, bad.actuarial_risk_score, s=3, c=ORANGE, alpha=.8, lw=0, rasterized=True)
    lim = 9
    ax.plot([0, lim], [0, lim], c=NAVY, lw=.8, ls="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("DS risk score"); ax.set_ylabel("Actuarial risk score")
    ax.text(.04, .94, f"Beyond tolerance: {bad.shape[0]:,} ({len(bad) / len(d):.1%})", transform=ax.transAxes,
            color=ORANGE, fontsize=7.5, weight="bold", va="top")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def delta_hist(d, path):
    fig, ax = plt.subplots(figsize=(3.55, 3.4), dpi=220)
    x = d.delta.clip(-3, 3)
    ax.hist(x[~d.beyond_tolerance], bins=90, range=(-3, 3), color=GREY, alpha=.9, label="within tolerance")
    ax.hist(x[d.beyond_tolerance], bins=90, range=(-3, 3), color=ORANGE, alpha=.95, label="beyond tolerance")
    ax.axvline(0, c=NAVY, lw=.8, ls="--")
    ax.set_xlabel("Δ = DS − Actuarial (clipped at ±3)"); ax.set_ylabel("Members")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    s = (f"median Δ  {d.delta.median():+.3f}\nMAE  {d.abs_delta.mean():.3f}\n"
         f"p95 |Δ|  {d.abs_delta.quantile(.95):.3f}\nbeyond tol.  {d.beyond_tolerance.mean():.1%}")
    ax.text(.97, .95, s, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=NAVY,
            bbox=dict(boxstyle="round,pad=.35", fc=LIGHT, ec=GREY, lw=.5))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def segment_bars(tables: dict, overall: float, path):
    titles = {"completeness_band": "Claims completeness", "utilization_band": "Utilization",
              "enrollment_band": "Months enrolled"}
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.35), dpi=220, sharey=True)
    for ax, (k, t) in zip(axes, tables.items()):
        v = t["pct_beyond_tolerance"] * 100
        cols = [ORANGE if x > overall * 100 * 1.5 else TEAL for x in v]
        ax.bar([str(i) for i in v.index], v, color=cols, width=.65)
        ax.axhline(overall * 100, c=NAVY, lw=.8, ls="--")
        for i, x in enumerate(v):
            ax.text(i, x + .5, f"{x:.1f}%", ha="center", fontsize=6.8, color=NAVY, bbox=dict(fc="white", ec="none", pad=.6))
        ax.set_title(titles[k], fontsize=8, color=NAVY, weight="bold")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("% beyond tolerance")
    axes[0].text(0.02, overall * 100 + .4, "overall", transform=axes[0].get_yaxis_transform(), fontsize=6.5, color=NAVY, va="bottom")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def draw_flow(steps, path, cols=1, figsize=(3.2, 5.2), highlight=(), pair=None):
    """Boxes joined by arrows. `pair=(i, j)` draws steps i and j side by side as one stage."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rows = int(np.ceil(n / cols))
    pos = []
    for i in range(n):
        r, c = divmod(i, cols)
        if rows > 1 or cols == 1:
            y = 1 - (r + .5) / rows
        x = (c + .5) / cols if cols > 1 else .5
        pos.append((x, y))
    bw, bh = (0.9 / cols if cols > 1 else .82), (.62 / rows)
    for i, ((x, y), label) in enumerate(zip(pos, steps)):
        fc = ORANGE if i in highlight else (TEAL if i in (pair or ()) else LIGHT)
        tc = "white" if i in highlight or i in (pair or ()) else NAVY
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc=fc, ec=NAVY if fc == LIGHT else fc, lw=.6))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.4, color=tc, wrap=True)
        if i < n - 1:
            x2, y2 = pos[i + 1]
            if y2 < y - 1e-6 and abs(x2 - x) < 1e-6:
                ax.annotate("", (x2, y2 + bh / 2), (x, y - bh / 2), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=.8))
            elif abs(y2 - y) < 1e-6:
                ax.annotate("", (x2 - bw / 2, y2), (x + bw / 2, y), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=.8))
            else:  # wrap to next row: elbow connector (down, across, down)
                ymid = (y - bh / 2 + y2 + bh / 2) / 2
                ax.plot([x, x, x2], [y - bh / 2, ymid, ymid], c=NAVY, lw=.8, solid_capstyle="butt")
                ax.annotate("", (x2, y2 + bh / 2), (x2, ymid), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=.8))
    fig.savefig(path, bbox_inches="tight", pad_inches=.03); plt.close(fig)


def system_flow(path):
    """Page-1 diagram with the DS <-> Actuarial pair drawn side by side."""
    fig, ax = plt.subplots(figsize=(3.3, 5.0), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    labels = ["Member + utilization data", "Data validation", None, "Reconciliation layer",
              "Root-cause classification", "Prioritized investigation queue",
              "Monitoring / downstream decision support"]
    n = len(labels); bh = .082
    ys = [1 - (i + .5) / n for i in range(n)]
    arrow = dict(arrowstyle="-|>", color=NAVY, lw=.8)

    def box(x, y, w, text, fc, tc, fs=7.2):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - bh / 2), w, bh,
                                    boxstyle="round,pad=0.004,rounding_size=0.02",
                                    fc=fc, ec=NAVY if fc == LIGHT else fc, lw=.6))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc)

    for i, (lab, y) in enumerate(zip(labels, ys)):
        if lab is None:
            box(.225, y, .43, "DS risk score", TEAL, "white")
            box(.775, y, .43, "Actuarial risk score", NAVY, "white")
            ax.annotate("", (.545, y), (.455, y), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2))
            ax.text(.5, y + .018, "Δ", ha="center", va="bottom", fontsize=9, color=ORANGE, weight="bold")
        else:
            hot = i in (4, 5)
            box(.5, y, .9, lab, ORANGE if hot else LIGHT, "white" if hot else NAVY,
                fs=6.6 if i == 6 else 7.2)

    pair_y = ys[2]
    for x in (.225, .775):                      # validation -> both scores, both scores -> reconciliation
        ax.annotate("", (x, pair_y + bh / 2), (.5 + (x - .5) * .35, ys[1] - bh / 2), arrowprops=arrow)
        ax.annotate("", (.5 + (x - .5) * .35, ys[3] + bh / 2), (x, pair_y - bh / 2), arrowprops=arrow)
    for i in (0, 3, 4, 5):
        ax.annotate("", (.5, ys[i + 1] + bh / 2), (.5, ys[i] - bh / 2), arrowprops=arrow)
    fig.savefig(path, bbox_inches="tight", pad_inches=.03); plt.close(fig)


def build_all(classified, seg_tables, fig_dir):
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    scatter_scores(classified, fig_dir / "fig1_scatter.png")
    delta_hist(classified, fig_dir / "fig2_delta.png")
    segment_bars(seg_tables, classified.beyond_tolerance.mean(), fig_dir / "fig3_segments.png")
    system_flow(fig_dir / "flow_system.png")
    draw_flow(["Raw snapshots", "Schema checks", "Completeness checks", "Scoring tables",
               "Reconciliation tests", "Anomaly detection", "Investigation queue", "Monitoring"],
              fig_dir / "flow_dependable.png", cols=4, figsize=(7.2, 1.7), highlight=(6,))
