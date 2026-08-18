#!/usr/bin/env python3
"""Figures for the four displays. Writes PDFs into <outdir>/figures/:
  cd_<condition>.pdf      critical-difference diagram (Friedman + Nemenyi)
  degradation_bars.pdf    per-suite mean acc degradation with bootstrap CIs
  reliability.pdf         pooled reliability curves ID vs OOD per model
  budget_curves.pdf       GBDT accuracy vs tuning budget, TFM reference lines
Usage: python src/make_figures.py <outdir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cd_diagram(ranks: pd.Series, cd: float, title, path):
    """Minimal CD diagram: models on a rank axis, CD bar, groups joined."""
    ranks = ranks.sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.35 * len(ranks)))
    lo, hi = ranks.min() - 0.3, ranks.max() + 0.3
    ax.set_xlim(lo, hi); ax.set_ylim(-1.5, len(ranks))
    ax.spines[["left", "right", "bottom"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.xaxis.tick_top(); ax.set_title(title, pad=28)
    for i, (m, r) in enumerate(ranks.items()):
        ax.plot([r, r], [len(ranks) - i - 1, len(ranks) - 0.2], "k-", lw=0.8)
        ax.text(r, len(ranks) - i - 1.15, f"{m} ({r:.2f})",
                ha="center", va="top", fontsize=8)
    if cd:
        ax.plot([lo + 0.05, lo + 0.05 + cd], [-1.2, -1.2], "k-", lw=2)
        ax.text(lo + 0.05 + cd / 2, -1.45, f"CD = {cd:.2f}", ha="center",
                fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def cd_value(k, n, alpha=0.05):
    """Nemenyi critical difference (Demsar 2006): q_alpha * sqrt(k(k+1)/6N)."""
    from scipy.stats import studentized_range
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def reliability_curve(y, p1, bins=12):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p1 >= lo) & (p1 < hi) if hi < 1 else (p1 >= lo) & (p1 <= hi)
        if m.sum() >= 20:
            xs.append(p1[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def main(outdir: Path):
    fig_dir = outdir / "figures"; fig_dir.mkdir(exist_ok=True)
    res = pd.read_csv(outdir / "aggregate" / "results.csv")
    ranks = pd.read_csv(outdir / "stats" / "ranks.csv")
    omnibus = json.loads((outdir / "stats" / "omnibus.json").read_text())

    # CD diagrams per condition
    for cond, g in ranks.groupby("condition"):
        ob = omnibus.get(cond, {})
        if "p_value" not in ob:
            continue
        cd = cd_value(ob["n_models"], ob["n_datasets"])
        cd_diagram(g.set_index("model")["mean_rank"], cd,
                   f"Mean ranks — {cond} (Friedman p={ob['p_value']:.2g})",
                   fig_dir / f"cd_{cond}.pdf")

    # degradation bars with bootstrap CIs
    boot_path = outdir / "stats" / "bootstrap.json"
    boot = json.loads(boot_path.read_text()) if boot_path.exists() else {}
    if boot:
        suites = sorted(boot)
        models = sorted({m for s in boot.values() for m in s})
        fig, axes = plt.subplots(1, len(suites), figsize=(4 * len(suites), 3.2),
                                 sharey=True, squeeze=False)
        for ax, suite in zip(axes[0], suites):
            ms = [m for m in models if m in boot[suite]]
            mu = [boot[suite][m]["mean"] for m in ms]
            err = np.array([[boot[suite][m]["mean"] - boot[suite][m]["ci95"][0],
                             boot[suite][m]["ci95"][1] - boot[suite][m]["mean"]]
                            for m in ms]).T
            ax.bar(range(len(ms)), mu, yerr=err, capsize=3)
            ax.set_xticks(range(len(ms)), ms, rotation=45, ha="right",
                          fontsize=8)
            ax.set_title(suite); ax.axhline(0, c="k", lw=0.5)
        axes[0][0].set_ylabel("acc(ID) − acc(OOD)")
        fig.tight_layout(); fig.savefig(fig_dir / "degradation_bars.pdf")
        plt.close(fig)

    # pooled reliability diagrams from saved predictions
    preds_dir = outdir / "preds"
    if preds_dir.exists():
        pool = {}
        for p in preds_dir.glob("*.npz"):
            model = p.stem.split("__")[1]
            z = np.load(p)
            if z["proba_id"].shape[1] != 2:
                continue
            d = pool.setdefault(model, {"id": ([], []), "ood": ([], [])})
            d["id"][0].append(z["proba_id"][:, 1].astype(float))
            d["id"][1].append(z["y_id"])
            if "proba_ood" in z.files:
                d["ood"][0].append(z["proba_ood"][:, 1].astype(float))
                d["ood"][1].append(z["y_ood"])
        if pool:
            models = sorted(pool)
            fig, axes = plt.subplots(1, len(models),
                                     figsize=(2.6 * len(models), 2.9),
                                     sharey=True, squeeze=False)
            for ax, m in zip(axes[0], models):
                ax.plot([0, 1], [0, 1], "k--", lw=0.6)
                for split, style in [("id", "-o"), ("ood", "-s")]:
                    ps, ys = pool[m][split]
                    if ps:
                        x, y = reliability_curve(np.concatenate(ys).astype(float),
                                                 np.concatenate(ps))
                        ax.plot(x, y, style, ms=3, label=split.upper())
                ax.set_title(m, fontsize=9); ax.set_xlabel("confidence")
            axes[0][0].set_ylabel("empirical accuracy")
            axes[0][0].legend(fontsize=7)
            fig.tight_layout(); fig.savefig(fig_dir / "reliability.pdf")
            plt.close(fig)

    # budget curves
    bc = pd.read_csv(outdir / "aggregate" / "budget_curves.csv")
    if len(bc):
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for (model, split), g in bc.groupby(["model", "split"]):
            curve = g.groupby("budget")["acc"].mean()
            ax.plot(curve.index, curve.values,
                    "-o" if split == "id" else "--s", ms=3,
                    label=f"{model} ({split})")
        for model, g in res[(res.kind == "tfm") & (res.split == "id")] \
                .groupby("model"):
            ax.axhline(g["acc"].mean(), ls=":", lw=1)
            ax.text(0.1, g["acc"].mean(), model, fontsize=7, va="bottom")
        ax.set_xlabel("tuning budget (trials)"); ax.set_ylabel("accuracy")
        ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(fig_dir / "budget_curves.pdf")
        plt.close(fig)

    print(f"[figures] wrote -> {fig_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
