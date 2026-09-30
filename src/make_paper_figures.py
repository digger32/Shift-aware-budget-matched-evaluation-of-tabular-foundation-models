#!/usr/bin/env python3
"""Publication figures for the manuscript, built from the frozen certified run.

Conventions applied (publication checklist):
  - vector PDF, sized to the IEEE single-column width (3.5 in), no shrink-to-fit;
  - TeX Gyre Termes so the figure font matches the Times body text;
  - 7-8 pt type, nothing smaller than the caption;
  - greyscale-safe: every series is distinguished by marker AND line style, not
    by colour alone; the palette is the Okabe-Ito colourblind-safe set;
  - no chartjunk: no 3D, no gradients, no shadows, no decorative frames.

Usage: python make_paper_figures.py <run_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import studentized_range

COL = 3.5                      # IEEE column width in inches
OKABE = {                      # colourblind-safe (Okabe-Ito)
    "tabicl": "#0072B2", "tabpfn2": "#D55E00", "tabpfn25": "#009E73",
    "xgb": "#000000", "lgbm": "#56B4E9", "catb": "#CC79A7",
    "logreg": "#999999", "mlp": "#E69F00",
}
MARK = {"tabicl": "o", "tabpfn2": "s", "tabpfn25": "^", "xgb": "D",
        "lgbm": "v", "catb": "P", "logreg": "x", "mlp": "*"}
LS = {"tabicl": "-", "tabpfn2": "--", "tabpfn25": "-.", "xgb": ":",
      "lgbm": (0, (3, 1, 1, 1)), "catb": (0, (5, 2)), "logreg": (0, (1, 1)),
      "mlp": (0, (4, 1, 1, 1, 1, 1))}
LABEL = {"tabicl": "TabICL", "tabpfn2": "TabPFN v2", "tabpfn25": "TabPFN 2.5",
         "xgb": "XGBoost", "lgbm": "LightGBM", "catb": "CatBoost",
         "logreg": "LogReg", "mlp": "MLP"}
TFM = ["tabicl", "tabpfn2", "tabpfn25"]


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    })


def cd_value(k, n, alpha=0.05):
    """Nemenyi critical difference for k models over n datasets."""
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def cd_panel(ax, ranks: pd.Series, cd: float, title: str):
    """Demsar-style critical-difference diagram: best rank on the left, model
    spurs alternating to the two margins, cliques of models that the Nemenyi
    test does not separate drawn as thick bars beneath the axis."""
    ranks = ranks.sort_values()
    models, vals = list(ranks.index), ranks.values
    n = len(models)
    lo, hi = np.floor(vals.min() * 2) / 2, np.ceil(vals.max() * 2) / 2
    span = hi - lo
    half = int(np.ceil(n / 2))
    row0, step = -1.05, 0.58
    ax.set_xlim(lo - 0.62 * span, hi + 0.62 * span)
    ax.set_ylim(row0 - step * (half - 1) - 0.55, 1.55)
    ax.axis("off")

    # rank axis, ticks and title
    ax.plot([lo, hi], [0, 0], color="black", lw=0.7)
    for tk in np.arange(lo, hi + 0.01, 1.0):
        ax.plot([tk, tk], [0, 0.14], color="black", lw=0.7)
        ax.text(tk, 0.20, f"{tk:.0f}", ha="center", va="bottom", fontsize=7)
    ax.text((lo + hi) / 2, 1.18, title, ha="center", va="bottom", fontsize=8)

    # critical-difference bar
    ybar = 0.74
    ax.plot([lo, lo + cd], [ybar, ybar], color="black", lw=1.4,
            solid_capstyle="butt")
    for x in (lo, lo + cd):
        ax.plot([x, x], [ybar - 0.08, ybar + 0.08], color="black", lw=1.4)
    ax.text(lo + cd / 2, ybar + 0.10, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=7)

    # model spurs
    for i, m in enumerate(models):
        r = vals[i]
        if i < half:
            y = row0 - step * i
            ax.plot([r, r], [0, y], color="black", lw=0.6)
            ax.plot([r, lo - 0.06 * span], [y, y], color="black", lw=0.6)
            ax.text(lo - 0.09 * span, y, f"{LABEL[m]} ({r:.2f})", ha="right",
                    va="center", fontsize=7)
        else:
            y = row0 - step * (n - 1 - i)
            ax.plot([r, r], [0, y], color="black", lw=0.6)
            ax.plot([r, hi + 0.06 * span], [y, y], color="black", lw=0.6)
            ax.text(hi + 0.09 * span, y, f"({r:.2f}) {LABEL[m]}", ha="left",
                    va="center", fontsize=7)

    # maximal cliques within the critical difference
    groups = []
    for i in range(n):
        j = i
        while j + 1 < n and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            groups.append((vals[i], vals[j]))
    kept = [g for k, g in enumerate(groups)
            if not any(g[0] >= h[0] and g[1] <= h[1]
                       for l, h in enumerate(groups) if l != k)]
    for k, (a, b) in enumerate(kept):
        ax.plot([a, b], [-0.28 - 0.20 * k, -0.28 - 0.20 * k], color="black",
                lw=2.4, solid_capstyle="round")


def fig_cd(run: Path, out: Path):
    om = json.load(open(run / "stats" / "omnibus.json"))
    ranks = pd.read_csv(run / "stats" / "ranks.csv")
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.15))
    for ax, cond, name in [(axes[0], "id", "in distribution"),
                           (axes[1], "ood", "out of distribution")]:
        r = ranks[ranks.condition == cond].set_index("model")["mean_rank"]
        o = om[cond]
        cd = cd_value(o["n_models"], o["n_datasets"])
        p = o["p_value"]
        if p < 1e-3:
            mant, ex = f"{p:.1e}".split("e")
            ptxt = f"$p = {mant}\\times 10^{{{int(ex)}}}$"
        else:
            ptxt = f"$p = {p:.2f}$"
        cd_panel(ax, r, cd, f"{name}: {o['n_datasets']} datasets, Friedman {ptxt}")
    fig.subplots_adjust(hspace=0.38)
    fig.savefig(out / "fig_cd.pdf")
    plt.close(fig)


def fig_calibration(run: Path, out: Path):
    df = pd.read_csv(run / "aggregate" / "results.csv")
    shift = df[df.dataset.isin(df[df.split == "ood"].dataset.unique())]
    p = shift.groupby(["model", "split"])[["acc", "ece", "smece"]].mean()
    ece_id, ece_ood = p.xs("id", level=1).ece, p.xs("ood", level=1).ece
    sm_id, sm_ood = p.xs("id", level=1).smece, p.xs("ood", level=1).smece
    acc_id, acc_ood = p.xs("id", level=1).acc, p.xs("ood", level=1).acc
    models = list(ece_id.sort_values().index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL, 2.05))

    # (a) ECE in distribution -> out of distribution, per model
    for m in models:
        ax1.plot([0, 1], [ece_id[m], ece_ood[m]], ls=LS[m], color=OKABE[m],
                 marker=MARK[m], ms=3, mew=0.8, lw=0.9)
    ax1.set_xlim(-0.25, 1.25)
    ax1.set_xticks([0, 1], ["ID", "OOD"])
    ax1.set_ylabel("expected calibration error")
    ax1.set_title("(a) calibration under shift", fontsize=8)
    ax1.grid(axis="y", lw=0.3, alpha=0.5)

    # (b) growth in calibration error against relative accuracy loss
    for m in models:
        x = 100 * (acc_id[m] - acc_ood[m]) / acc_id[m]
        y = ece_ood[m] / ece_id[m]
        ax2.plot(x, y, marker=MARK[m], color=OKABE[m], ms=4, mew=0.9, ls="none",
                 label=LABEL[m])
        ax2.plot(x, sm_ood[m] / sm_id[m], marker=MARK[m], mfc="none",
                 mec=OKABE[m], ms=5, mew=0.7, ls="none")
    ax2.axhline(1.0, color="black", lw=0.6, ls="--")
    ax2.set_xlabel("accuracy loss (%)")
    ax2.set_ylabel("calibration error growth ($\\times$)")
    ax2.set_title("(b) rates of degradation", fontsize=8)
    ax2.grid(lw=0.3, alpha=0.5)
    ax2.set_xlim(0, 8.6)
    ax2.set_ylim(0.98, 1.9)
    ax2.text(0.25, 1.03, "no change", fontsize=6, va="bottom")

    handles, labels = ax2.get_legend_handles_labels()
    fig.tight_layout(pad=0.3, rect=(0, 0.17, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.0)
    fig.savefig(out / "fig_calibration.pdf")
    plt.close(fig)


def fig_budget(run: Path, out: Path):
    bc = pd.read_csv(run / "aggregate" / "budget_curves.csv")
    df = pd.read_csv(run / "aggregate" / "results.csv")
    shift_ds = df[df.split == "ood"].dataset.unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL, 1.95), sharex=True)
    for ax, split, subset, name in [
            (ax1, "id", None, "(a) in distribution, 17 datasets"),
            (ax2, "ood", shift_ds, "(b) out of distribution, 7 datasets")]:
        sel = bc[bc.split == split]
        for m in ["xgb", "lgbm", "catb"]:
            c = sel[sel.model == m].groupby("budget").acc.mean()
            ax.plot(c.index, c.values, ls=LS[m], color=OKABE[m], marker=MARK[m],
                    ms=3, mew=0.8, label=LABEL[m])
        d = df[df.split == split]
        if subset is not None:
            d = d[d.dataset.isin(subset)]
        # reference lines for the zero-tuning foundation models; models whose
        # means coincide share a single label so the annotations cannot overlap
        vals = {m: d[d.model == m].acc.mean() for m in TFM}
        for m, v in vals.items():
            ax.axhline(v, color=OKABE[m], ls=LS[m], lw=0.9)
        groups = {}
        for m, v in sorted(vals.items(), key=lambda kv: kv[1]):
            key = next((k for k in groups if abs(k - v) < 2e-4), v)
            groups.setdefault(key, []).append(m)
        for v, ms in groups.items():
            ax.annotate(" / ".join(LABEL[m] for m in ms), xy=(30, v),
                        xytext=(-1, 1.6), textcoords="offset points",
                        ha="right", va="bottom", fontsize=6,
                        color=OKABE[ms[0]])
        ax.margins(y=0.14)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("random-search trials")
        ax.set_xticks([0, 5, 10, 20, 30])
        ax.grid(lw=0.3, alpha=0.5)
    ax1.set_ylabel("accuracy")
    ax1.legend(loc="lower right", fontsize=6.5, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_budget.pdf")
    plt.close(fig)


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig_cd(run, out)
    fig_calibration(run, out)
    fig_budget(run, out)
    print(f"[figures] wrote fig_cd.pdf, fig_calibration.pdf, fig_budget.pdf -> {out}")


if __name__ == "__main__":
    main()
