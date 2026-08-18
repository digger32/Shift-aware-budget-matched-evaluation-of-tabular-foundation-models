#!/usr/bin/env python3
"""Review-proofing gate for the shift-aware evaluation protocol.

Reads run_meta.json + manifest.jsonl + per-unit JSONs and asserts the
conditions that keep dirty numbers out of figures. NON-ZERO exit blocks the
report stage in pipeline.sh.

Assertions:
  A1  clean final run       resume DISABLED, no skips, single run_started
  B1  external validity     every comparative claim has >=1 independent-dataset run
  C1  calibration present   'ece' recorded (enable: require_calibration)
  D1  optimism gap          recorded for tuned configs (require_optimism_gap)
  E1  stats present         omnibus + posthoc artifacts exist (require_stats)
  F1  subsample freeze      per (dataset, seed) ALL models share identical
                            train/id/ood subsample hashes (require_subsample_freeze)
  F2  device policy         each unit ran on the device its model is pinned to
                            (device_policy map in config)
  F3  tuning log            every unit of the listed models carries a full HPO
                            log with the declared n_trials (require_tuning_log_for)
  F4  all units ok          manifest contains no fail/timeout entries
                            (require_all_ok — final pass only)

Usage: python src/review_gate.py <outdir> [--config configs/gate_config.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[gate] PyYAML required: pip install pyyaml --break-system-packages")
    sys.exit(2)


def load_manifest(outdir: Path):
    mf = outdir / "manifest.jsonl"
    if not mf.exists():
        return []
    return [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]


def load_units(outdir: Path):
    units = []
    for p in outdir.glob("*__*__seed*.json"):
        try:
            units.append(json.loads(p.read_text()))
        except Exception:
            pass
    return units


def check_A1(outdir, manifest, cfg):
    meta_path = outdir / "run_meta.json"
    if not meta_path.exists():
        return False, "run_meta.json missing — cannot verify the final pass"
    meta = json.loads(meta_path.read_text())
    if not meta.get("no_resume", False):
        return False, "final pass ran WITHOUT --no-resume (resume was enabled)"
    if any(r.get("status") == "skip" for r in manifest):
        return False, "manifest shows skipped units in a no-resume pass"
    run_started = meta.get("run_started")
    stale = [r["unit"] for r in manifest if r.get("started") != run_started]
    if stale:
        return False, (f"{len(stale)} unit(s) carry a different run_started "
                       f"(carry-over): {stale[:3]}...")
    return True, "final pass clean: --no-resume, no skips, single run_started"


def check_B1(outdir, units, cfg):
    claims = cfg.get("comparative_claims", [])
    if not claims:
        return False, "no comparative_claims declared in config — declare them"
    present = {u.get("dataset") for u in units}
    failures = []
    for c in claims:
        if c.get("waive"):
            if not c.get("waiver_justification"):
                failures.append(f"claim '{c.get('id')}' waived without justification")
            continue
        needed = set(c.get("independent_datasets", []))
        if not needed:
            failures.append(f"claim '{c.get('id')}' lists no independent_datasets")
        elif not (needed & present):
            failures.append(f"claim '{c.get('id')}' has no run on any of {sorted(needed)}")
    if failures:
        return False, "; ".join(failures)
    return True, f"{len(claims)} comparative claim(s) each have an independent-dataset run"


def check_metric_present(units, key, label):
    have = [u for u in units if (u.get("metrics") or {}).get(key) is not None]
    if not have:
        return False, f"no unit recorded '{key}' ({label})"
    return True, f"'{key}' present in {len(have)} unit(s) ({label})"


def check_E1_stats(outdir, cfg):
    art = cfg.get("stats_artifacts", ["stats/omnibus.json", "stats/posthoc.json"])
    missing = [a for a in art if not (outdir / a).exists() and not Path(a).exists()]
    if missing:
        return False, f"missing stats artifacts: {missing}"
    return True, f"stats artifacts present: {art}"


def check_F1_subsample(units):
    groups = {}
    for u in units:
        sub = u.get("subsample") or {}
        groups.setdefault((u["dataset"], u["seed"]), set()).add(
            (sub.get("train_hash"), sub.get("id_hash"), sub.get("ood_hash")))
    bad = [k for k, v in groups.items() if len(v) != 1 or (None, None, None) in v]
    if bad:
        return False, f"subsample hashes differ or missing for {bad[:3]}..."
    return True, f"identical frozen subsample across models for {len(groups)} (dataset, seed) cells"


def check_F2_device(units, policy):
    bad = [f"{u['dataset']}__{u['model']}__seed{u['seed']}"
           for u in units
           if u["model"] in policy and u.get("device") != policy[u["model"]]]
    if bad:
        return False, f"device policy violated: {bad[:3]}..."
    return True, f"device policy held for {len(units)} unit(s)"


def check_F3_tuning(units, models, n_trials):
    bad = []
    for u in units:
        if u["model"] not in models:
            continue
        t = u.get("tuning") or {}
        if t.get("n_trials") != n_trials or t.get("wall_s") is None:
            bad.append(f"{u['dataset']}__{u['model']}__seed{u['seed']}")
    if bad:
        return False, f"tuning log missing/short for {bad[:3]}..."
    n = sum(1 for u in units if u["model"] in models)
    return True, f"full {n_trials}-trial HPO log present for {n} tuned unit(s)"


def check_F4_all_ok(manifest):
    bad = [r["unit"] for r in manifest if r.get("status") != "ok"]
    if bad:
        return False, f"{len(bad)} unit(s) not ok in manifest: {bad[:3]}..."
    return True, f"all {len(manifest)} manifest entries ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--config", default="gate_config.yaml")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    cfg_path = Path(a.config)
    if not cfg_path.exists():
        cfg_path = outdir / a.config
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    manifest = load_manifest(outdir)
    units = load_units(outdir)

    results = [("A1 clean-final-run", *check_A1(outdir, manifest, cfg)),
               ("B1 external-validity", *check_B1(outdir, units, cfg))]
    if cfg.get("require_calibration"):
        results.append(("C1 calibration", *check_metric_present(
            units, "ece", "calibration present")))
    if cfg.get("require_optimism_gap"):
        results.append(("D1 optimism-gap", *check_metric_present(
            units, "optimism_gap", "HPO honesty")))
    if cfg.get("require_stats"):
        results.append(("E1 stats", *check_E1_stats(outdir, cfg)))
    if cfg.get("require_subsample_freeze"):
        results.append(("F1 subsample-freeze", *check_F1_subsample(units)))
    if cfg.get("device_policy"):
        results.append(("F2 device-policy", *check_F2_device(
            units, cfg["device_policy"])))
    if cfg.get("require_tuning_log_for"):
        results.append(("F3 tuning-log", *check_F3_tuning(
            units, cfg["require_tuning_log_for"], cfg.get("tuning_n_trials", 30))))
    if cfg.get("require_all_ok"):
        results.append(("F4 all-units-ok", *check_F4_all_ok(manifest)))

    print("=" * 64)
    print(f"REVIEW-PROOFING GATE  | outdir={outdir}")
    print("=" * 64)
    ok = True
    for name, passed, msg in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name:24s} {msg}")
        ok = ok and passed
    print("=" * 64)
    if not ok:
        print("GATE FAILED — do not freeze these numbers into figures.")
        sys.exit(1)
    print("GATE PASSED — numbers are clean to freeze.")


if __name__ == "__main__":
    main()
