#!/usr/bin/env bash
# Evaluation pipeline: one command per stage. All paths are relative to the
# project root, wherever the script is invoked from. Long stages are best run
# inside a terminal multiplexer:
#     tmux new -s eval ; ./pipeline.sh <stage> ; Ctrl-b d
#
# Stages:
#   prep      download EVERYTHING (datasets + checkpoints), then go offline
#   gate_test unit-test the gate (clean->0, dirty->1) — run before anything real
#   smoke     synthetic data, cpu-only models, full chain incl. gate, <2 min
#   micro     1 representative unit per model on the largest dataset per suite;
#             prints wall_s for extrapolation BEFORE committing to the grid
#   pilot     2 seeds x 2 datasets x all models
#   full      complete grid, resume ENABLED (runs/full)
#   final     fresh dir, --no-resume, gate MUST pass (runs/final_*)
#   report    aggregate + stats + gate + figures on an existing outdir ($2)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
R=src
CFG=configs/grid.yaml

# ---- environment ----------------------------------------------------------- #
# CUDA allocator: expandable segments prevent the fragmentation failure mode
# in which a large reserved-but-unallocated pool triggers out-of-memory errors
# under varying allocation sizes.
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"   # alias, older torch
# Offline after prep: nothing downloads mid-run.
if [[ "${1:-}" != "prep" && -f data/prepared/READY ]]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

PY=python3

chain () {  # chain <outdir> <gate_config>
    $PY $R/aggregate.py "$1"
    $PY $R/stats.py "$1"
    $PY $R/review_gate.py "$1" --config "$2"
    $PY $R/make_figures.py "$1"
}

case "${1:-}" in
  prep)
    $PY $R/data_prep.py --config $CFG
    $PY $R/download_models.py --config $CFG
    echo "[prep] compare configs/gate_config.yaml claims against data/prepared/registry.json NOW:"
    $PY - <<'PYEOF'
import json, yaml
reg = set(json.load(open("data/prepared/registry.json"))["datasets"])
cfg = yaml.safe_load(open("configs/gate_config.yaml"))
for c in cfg["comparative_claims"]:
    miss = [d for d in c.get("independent_datasets", []) if d not in reg]
    print(f"  claim {c['id']}: " + ("OK" if not miss else f"MISSING {miss} — edit configs/gate_config.yaml"))
PYEOF
    ;;
  gate_test)
    $PY tests/test_gate.py
    ;;
  smoke)
    $PY tests/test_gate.py
    $PY $R/data_prep.py --synthetic --data-dir data/prepared_smoke --config $CFG
    rm -rf runs/smoke
    $PY $R/bench_runner.py --outdir runs/smoke --data-dir data/prepared_smoke \
        --datasets all --models logreg,xgb --seeds 0 --no-resume --timeout-s 300
    chain runs/smoke tests/gate_config_smoke.yaml
    echo "[smoke] full chain incl. gate passed"
    ;;
  micro)
    # largest dataset per suite x every model x 1 seed -> read wall_s, extrapolate.
    UNITS=$($PY - <<'PYEOF'
import json
reg = json.load(open("data/prepared/registry.json"))["datasets"]
best = {}
for k, v in reg.items():
    s = v["suite"]
    if s not in best or v["n_train"] > reg[best[s]]["n_train"]:
        best[s] = k
print(",".join(best.values()))
PYEOF
)
    $PY $R/bench_runner.py --outdir runs/micro --datasets "$UNITS" --seeds 0
    echo "[micro] wall_s per unit (extrapolate BEFORE full — estimates run 3-4x low):"
    $PY - <<'PYEOF'
import json, yaml
rows = [json.loads(l) for l in open("runs/micro/manifest.jsonl")]
cfg = yaml.safe_load(open("configs/grid.yaml"))
reg = json.load(open("data/prepared/registry.json"))["datasets"]
per_model = {}
for r in rows:
    per_model.setdefault(r["model"], []).append(r["wall_s"])
n_units = len(reg) * len(cfg["seeds"])
total = 0
for m, ws in sorted(per_model.items()):
    w = max(ws)
    total += w * n_units
    print(f"  {m:12s} max {w:8.1f}s/unit -> ~{w*n_units/3600:6.1f} h for {n_units} units")
print(f"  GRID UPPER BOUND ~{total/3600:.1f} h (max-unit basis, sequential)")
PYEOF
    ;;
  pilot)
    PILOT=$($PY -c "import json; ks=sorted(json.load(open('data/prepared/registry.json'))['datasets']); print(','.join(ks[:2]))")
    $PY $R/bench_runner.py --outdir runs/pilot --datasets "$PILOT" --seeds 0,1
    chain runs/pilot configs/gate_config.yaml || true   # pilot: chain must RUN; A1 fails by design (resume on)
    ;;
  full)
    $PY $R/bench_runner.py --outdir runs/full --datasets all
    $PY $R/aggregate.py runs/full && $PY $R/stats.py runs/full && $PY $R/make_figures.py runs/full
    ;;
  final)
    RUNDIR="runs/final_$(date +%Y%m%d_%H%M)"
    $PY $R/bench_runner.py --outdir "$RUNDIR" --datasets all --no-resume
    chain "$RUNDIR" configs/gate_config.yaml
    echo "[final] GATE PASSED on $RUNDIR — these numbers may be frozen"
    ;;
  report)
    chain "${2:?usage: pipeline.sh report <outdir>}" configs/gate_config.yaml
    ;;
  *)
    echo "usage: ./pipeline.sh {prep|gate_test|smoke|micro|pilot|full|final|report <outdir>}"
    exit 2
    ;;
esac
