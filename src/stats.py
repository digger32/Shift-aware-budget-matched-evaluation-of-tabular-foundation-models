#!/usr/bin/env python3
"""Statistics for the three claims. Reads <outdir>/aggregate/results.csv and
writes the EXACT artifacts the gate's E1 check reads:
  <outdir>/stats/omnibus.json   Friedman per split condition (id, ood, delta)
  <outdir>/stats/posthoc.json   Nemenyi p-value matrices per condition
  <outdir>/stats/wilcoxon.json  per-suite paired Wilcoxon: each TFM vs best GBDT
  <outdir>/stats/bootstrap.json per-suite bootstrap CIs of mean degradation
  <outdir>/stats/ranks.csv      mean ranks per condition (input to CD diagram)
Primary metric: accuracy (auroc and ece analysed in the calibration figure).
Usage: python src/stats.py <outdir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss


def pivot(df, value):
    """dataset x model matrix of seed-mean `value`."""
    t = (df.groupby(["dataset", "model"])[value].mean().unstack("model"))
    return t.dropna(axis=0, how="any")


def friedman_block(mat: pd.DataFrame):
    if mat.shape[0] < 3 or mat.shape[1] < 3:
        return {"note": "insufficient for omnibus",
                "n_datasets": int(mat.shape[0]), "n_models": int(mat.shape[1])}
    stat, p = ss.friedmanchisquare(*[mat[c].values for c in mat.columns])
    return {"statistic": float(stat), "p_value": float(p),
            "n_datasets": int(mat.shape[0]), "n_models": int(mat.shape[1]),
            "models": list(mat.columns)}


def nemenyi_block(mat: pd.DataFrame):
    if mat.shape[0] < 3 or mat.shape[1] < 3:
        return {"note": "insufficient for posthoc"}
    import scikit_posthocs as sp
    pm = sp.posthoc_nemenyi_friedman(mat.values)
    pm.index = pm.columns = list(mat.columns)
    return {"p_matrix": pm.round(6).to_dict()}


def mean_ranks(mat: pd.DataFrame, higher_better=True):
    r = mat.rank(axis=1, ascending=not higher_better)
    return r.mean().sort_values().round(3).to_dict()


def wilcoxon_suite(df, tfm_models, gbdt_models, value):
    out = {}
    for suite, g in df.groupby("suite"):
        m = pivot(g, value)
        gb = [c for c in gbdt_models if c in m.columns]
        tf = [c for c in tfm_models if c in m.columns]
        if not gb or not tf or m.shape[0] < 3:
            out[suite] = {"note": "insufficient"}
            continue
        best_gbdt = max(gb, key=lambda c: m[c].mean())
        out[suite] = {"best_gbdt": best_gbdt}
        for t in tf:
            try:
                stat, p = ss.wilcoxon(m[t].values, m[best_gbdt].values)
                out[suite][t] = {"vs": best_gbdt, "statistic": float(stat),
                                 "p_value": float(p),
                                 "mean_delta": float((m[t] - m[best_gbdt]).mean())}
            except ValueError as e:
                out[suite][t] = {"note": str(e)}
    return out


def bootstrap_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "n": int(len(x))}


def main(outdir: Path):
    df = pd.read_csv(outdir / "aggregate" / "results.csv")
    stats_dir = outdir / "stats"
    stats_dir.mkdir(exist_ok=True)

    kinds = df.drop_duplicates("model").set_index("model")["kind"].to_dict()
    tfm = [m for m, k in kinds.items() if k == "tfm"]
    gbdt = [m for m, k in kinds.items() if k == "gbdt"]

    id_df = df[df.split == "id"]
    ood_df = df[df.split == "ood"]
    conditions = {"id": pivot(id_df, "acc")}
    if len(ood_df):
        conditions["ood"] = pivot(ood_df, "acc")
        deg = ood_df.dropna(subset=["acc_degradation"])
        conditions["degradation"] = pivot(deg, "acc_degradation")

    omnibus, posthoc, ranks_rows = {}, {}, []
    for name, mat in conditions.items():
        omnibus[name] = friedman_block(mat)
        posthoc[name] = nemenyi_block(mat)
        hb = name != "degradation"          # lower degradation is better
        for model, r in mean_ranks(mat, higher_better=hb).items():
            ranks_rows.append({"condition": name, "model": model, "mean_rank": r})

    (stats_dir / "omnibus.json").write_text(json.dumps(omnibus, indent=2))
    (stats_dir / "posthoc.json").write_text(json.dumps(posthoc, indent=2))
    (stats_dir / "wilcoxon.json").write_text(json.dumps({
        "id": wilcoxon_suite(id_df, tfm, gbdt, "acc"),
        "ood": wilcoxon_suite(ood_df, tfm, gbdt, "acc") if len(ood_df) else {},
    }, indent=2))

    boot = {}
    for (suite, model), g in ood_df.groupby(["suite", "model"]):
        x = g.dropna(subset=["acc_degradation"]) \
             .groupby("dataset")["acc_degradation"].mean()
        if len(x) >= 2:
            boot.setdefault(suite, {})[model] = bootstrap_ci(x.values)
    (stats_dir / "bootstrap.json").write_text(json.dumps(boot, indent=2))
    pd.DataFrame(ranks_rows).to_csv(stats_dir / "ranks.csv", index=False)
    print(f"[stats] wrote omnibus/posthoc/wilcoxon/bootstrap/ranks -> {stats_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
