#!/usr/bin/env python3
"""Gate unit test: fabricate a synthetic clean run and a synthetic dirty run,
invoke the gate on both, and prove exit codes 0 and 1 respectively.
Run from the project root:  python tests/test_gate.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "tests" / "gate_config_smoke.yaml"


def unit(dataset, model, seed, device, hashes, started):
    return {
        "unit_json": {
            "dataset": dataset, "suite": "tableshift", "model": model,
            "seed": seed, "device": device, "kind": "linear",
            "subsample": {"train_hash": hashes[0], "id_hash": hashes[1],
                          "ood_hash": hashes[2], "n_train": 10, "n_id": 5,
                          "n_ood": 5},
            "metrics": {"acc": 0.9, "ece": 0.05},
            "tuning": None, "budget_curve": [], "wall_s": 1.0,
        },
        "manifest": {"unit": f"{dataset}__{model}__seed{seed}",
                     "dataset": dataset, "model": model, "seed": seed,
                     "status": "ok", "started": started,
                     "finished": started, "wall_s": 1.0, "no_resume": True},
    }


def build_run(root: Path, clean: bool):
    if root.exists():
        shutil.rmtree(root)
    (root / "stats").mkdir(parents=True)
    started = "2026-07-19T00:00:00+00:00"
    (root / "run_meta.json").write_text(json.dumps(
        {"run_started": started, "no_resume": clean, "shard": "0/1"}))
    manifest = []
    for dataset in ["synth_shift", "synth_iid"]:
        for model in ["logreg", "xgb"]:
            h = ("aaa", "bbb", "ccc")
            if not clean and model == "xgb":
                h = ("XXX", "bbb", "ccc")            # subsample-freeze violation
            u = unit(dataset, model, 0, "cpu", h, started)
            (root / f"{dataset}__{model}__seed0.json").write_text(
                json.dumps(u["unit_json"]))
            manifest.append(u["manifest"])
    if not clean:
        manifest.append({"unit": "synth_shift__logreg__seed1", "status": "skip",
                         "started": started, "no_resume": False})
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest) + "\n")
    (root / "stats" / "omnibus.json").write_text("{}")
    if clean:
        (root / "stats" / "posthoc.json").write_text("{}")   # dirty: missing


def gate(root: Path) -> int:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "review_gate.py"), str(root),
         "--config", str(CFG)]).returncode


def main():
    tmp = ROOT / "tests" / "tmp"
    build_run(tmp / "clean", clean=True)
    build_run(tmp / "dirty", clean=False)
    rc_clean, rc_dirty = gate(tmp / "clean"), gate(tmp / "dirty")
    print(f"[test_gate] clean rc={rc_clean} (want 0), dirty rc={rc_dirty} (want 1)")
    if rc_clean != 0 or rc_dirty != 1:
        sys.exit("[test_gate] FAILED — fix the gate before any real run")
    print("[test_gate] PASSED")


if __name__ == "__main__":
    main()
