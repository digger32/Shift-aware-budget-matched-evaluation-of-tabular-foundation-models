#!/usr/bin/env python3
"""Merge per-unit JSONs into tidy CSVs under <outdir>/aggregate/:
  results.csv        one row per unit x split (id/ood) with all metrics
  budget_curves.csv  one row per GBDT unit x budget checkpoint x split
Usage: python src/aggregate.py <outdir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

SPLIT_METRICS = ["acc", "bacc", "auroc", "logloss", "brier", "ece", "smece"]


def main(outdir: Path):
    rows, curve_rows = [], []
    for p in sorted(outdir.glob("*__*__seed*.json")):
        u = json.loads(p.read_text())
        base = {"dataset": u["dataset"], "suite": u["suite"],
                "model": u["model"], "seed": u["seed"], "kind": u["kind"],
                "device": u["device"], "wall_s": u.get("wall_s")}
        m = u["metrics"]
        rows.append({**base, "split": "id",
                     **{k: m.get(k) for k in SPLIT_METRICS},
                     "optimism_gap": m.get("optimism_gap")})
        if "acc_ood" in m:
            rows.append({**base, "split": "ood",
                         **{k: m.get(f"{k}_ood") for k in SPLIT_METRICS},
                         "acc_degradation": m.get("acc_degradation")})
        for c in u.get("budget_curve") or []:
            for split in ("id", "ood"):
                if split in c:
                    curve_rows.append({**base, "budget": c["budget"],
                                       "split": split, **c[split]})

    agg = outdir / "aggregate"
    agg.mkdir(exist_ok=True)
    if not rows:
        sys.exit(f"[aggregate] no unit JSONs in {outdir}")
    pd.DataFrame(rows).to_csv(agg / "results.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(agg / "budget_curves.csv", index=False)
    print(f"[aggregate] {len(rows)} result rows, {len(curve_rows)} curve rows "
          f"-> {agg}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
