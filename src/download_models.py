#!/usr/bin/env python3
"""Pre-download every TFM checkpoint ONCE, so all run stages work offline
(HF_HUB_OFFLINE=1). Strategy: instantiate each classifier exactly the way the
worker will and fit a 20-row toy problem — that triggers the library's own
checkpoint download into its cache.

Verified against the pinned releases (tabpfn 8.1.0, tabicl 2.1.1):
  - tabpfn v2:   TabPFNClassifier.create_default_for_version(ModelVersion.V2)
                 -> HF Prior-Labs/TabPFN-v2-clf
  - tabpfn v2.5: ...(ModelVersion.V2_5), default ckpt is real-data finetuned
                 (RealTabPFN lineage) -> HF Prior-Labs/tabpfn_2_5, Prior Labs
                 License. If the repository asks for licence acceptance, log in
                 once with `hf auth login`, accept on the model page, and rerun.
  - tabicl:      TabICLClassifier() -> HF jingang/TabICL, package-default
                 checkpoint tabicl-classifier-v2-20260212.ckpt

Usage:  python src/download_models.py --config configs/grid.yaml
Writes <models_dir>/READY on success and per-model status lines.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np


def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def fetch_tabpfn(version_name):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    X, y = toy()
    clf = TabPFNClassifier.create_default_for_version(
        getattr(ModelVersion, version_name), device="cpu")
    clf.fit(X, y); clf.predict(X)
    return f"tabpfn {version_name} default checkpoint cached"


def fetch_tabicl():
    from tabicl import TabICLClassifier
    X, y = toy()
    TabICLClassifier(device="cpu").fit(X, y).predict(X)
    return "tabicl default checkpoint cached"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/grid.yaml")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text())
    models_dir = Path(cfg["runtime"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    status = {}
    for name, fn in [("tabpfn2", lambda: fetch_tabpfn("V2")),
                     ("tabpfn25", lambda: fetch_tabpfn("V2_5")),
                     ("tabicl", fetch_tabicl)]:
        try:
            status[name] = {"ok": True, "note": fn()}
            print(f"[models] {name}: ok", flush=True)
        except Exception as e:
            status[name] = {"ok": False, "note": f"{type(e).__name__}: {e}"}
            print(f"[models] {name}: FAILED — {e}", flush=True)
            traceback.print_exc()
    (models_dir / "status.json").write_text(json.dumps(status, indent=2))

    if not any(v["ok"] for v in status.values()):
        sys.exit("[models] no TFM checkpoint available — fix before running")
    (models_dir / "READY").write_text("ok\n")
    print("[models] READY written — TFM units can now run offline")


if __name__ == "__main__":
    main()
