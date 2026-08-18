#!/usr/bin/env python3
"""Job-based benchmark runner.

Units are INDEPENDENT (dataset x model x seed); each runs in its own
subprocess with a hard wall-clock timeout, stdout/stderr captured to
runs/<pass>/logs/<uid>.log. Resume skips units whose output exists; the FINAL
pass runs with --no-resume into a fresh outdir (the gate's A1 invariant).

Launch (a terminal multiplexer is recommended for long runs):
    tmux new -s eval
    python bench_runner.py --outdir runs/full --datasets all
    # detach: Ctrl-b d ; reattach: tmux attach -t t1full
Final:
    tmux new -s eval_final
    python bench_runner.py --outdir runs/final --datasets all --no-resume
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def unit_id(dataset, model, seed):
    return f"{dataset}__{model}__seed{seed}"


def append_manifest(outdir: Path, record: dict):
    with (outdir / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def worker_env(cfg):
    """Env for the unit subprocess: pinned threads for every math backend and
    the CUDA allocator setting that avoids the fragmentation failure mode."""
    env = os.environ.copy()
    t = str(cfg["tuning"]["threads"])
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"]:
        env[var] = t
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def run_worker(args, cfg):
    outdir = Path(args.outdir)
    out_path = outdir / f"{unit_id(args.dataset, args.model, args.seed)}.json"
    import unit_worker
    unit_worker.run_unit(args.dataset, args.model, args.seed, cfg, out_path,
                         sampler=args.sampler, data_dir=args.data_dir,
                         preds_dir=str(outdir / "preds"))


def run_orchestrator(args, cfg):
    outdir = Path(args.outdir)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir or cfg["runtime"]["data_dir"])
    if not (data_dir / "READY").exists():
        sys.exit(f"[runner] {data_dir}/READY missing — run data_prep first "
                 "(offline-first discipline: download everything, then run)")
    registry = json.loads((data_dir / "registry.json").read_text())["datasets"]

    datasets = (sorted(registry) if args.datasets == "all"
                else args.datasets.split(","))
    missing = [d for d in datasets if d not in registry]
    if missing:
        sys.exit(f"[runner] datasets not in registry: {missing}")
    models = (sorted(cfg["models"]) if args.models == "all"
              else args.models.split(","))
    tfm_models = [m for m in models if cfg["models"][m]["kind"] == "tfm"]
    if tfm_models and not Path(cfg["runtime"]["models_dir"], "READY").exists():
        sys.exit("[runner] models/READY missing — run download_models first")
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else cfg["seeds"])
    timeout_s = args.timeout_s or cfg["runtime"]["timeout_s"]

    run_started = datetime.now(timezone.utc).isoformat()
    (outdir / "run_meta.json").write_text(json.dumps({
        "run_started": run_started, "no_resume": args.no_resume,
        "shard": args.shard, "sampler": args.sampler,
        "axes": {"datasets": datasets, "models": models, "seeds": seeds},
        "timeout_s": timeout_s, "config": args.config,
    }, indent=2))

    units = [(d, m, s) for d in datasets for m in models for s in seeds]
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        if args.no_resume:
            sys.exit("[runner] refuse: --no-resume final pass must run on ONE box "
                     "(--shard 0/1); shard only dev sweeps.")
        units = [u for i, u in enumerate(units) if i % shard_n == shard_k]
        print(f"[runner] shard {shard_k}/{shard_n}: {len(units)} units this box",
              flush=True)
    print(f"[runner] {len(units)} units | outdir={outdir} | "
          f"no_resume={args.no_resume} | timeout={timeout_s}s", flush=True)

    env = worker_env(cfg)
    n_done = n_skip = n_fail = n_timeout = 0
    for d, m, s in units:
        uid = unit_id(d, m, s)
        out_path = outdir / f"{uid}.json"
        if out_path.exists() and not args.no_resume:
            n_skip += 1
            print(f"[skip] {uid} (output exists)", flush=True)
            continue
        if out_path.exists() and args.no_resume:
            out_path.unlink()

        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--dataset", d, "--model", m, "--seed", str(s),
               "--outdir", str(outdir), "--config", args.config,
               "--sampler", args.sampler]
        if args.data_dir:
            cmd += ["--data-dir", args.data_dir]
        t0 = time.time()
        status = "ok"
        log_path = outdir / "logs" / f"{uid}.log"
        with log_path.open("w") as log:
            try:
                subprocess.run(cmd, timeout=timeout_s, check=True,
                               stdout=log, stderr=subprocess.STDOUT, env=env)
            except subprocess.TimeoutExpired:
                status = "timeout"; n_timeout += 1
                print(f"[TIMEOUT] {uid} > {timeout_s}s — unit killed, batch "
                      f"continues | log: {log_path}", flush=True)
            except subprocess.CalledProcessError as e:
                status = f"fail(rc={e.returncode})"; n_fail += 1
                print(f"[FAIL] {uid} rc={e.returncode} — batch continues | "
                      f"log: {log_path}", flush=True)
            else:
                if not out_path.exists():
                    status = "fail(no_output)"; n_fail += 1
                else:
                    n_done += 1
                    print(f"[ok] {uid} ({time.time()-t0:.1f}s)", flush=True)

        append_manifest(outdir, {
            "unit": uid, "dataset": d, "model": m, "seed": s,
            "status": status, "started": run_started,
            "finished": datetime.now(timezone.utc).isoformat(),
            "wall_s": round(time.time() - t0, 1), "no_resume": args.no_resume,
        })

    print(f"[runner] done | ok={n_done} skip={n_skip} fail={n_fail} "
          f"timeout={n_timeout}", flush=True)
    if n_fail or n_timeout:
        print("[runner] some units did not complete — inspect logs/manifest "
              "before freezing.", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="internal: one unit")
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--datasets", default="all", help="'all' = realised registry")
    ap.add_argument("--models", default="all")
    ap.add_argument("--seeds", default=None, help="default: seeds from config")
    ap.add_argument("--sampler", default=None,
                    help="override GBDT HPO sampler (random|tpe)")
    ap.add_argument("--outdir", default="runs/dev")
    ap.add_argument("--data-dir", dest="data_dir", default=None)
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--no-resume", dest="no_resume", action="store_true")
    ap.add_argument("--dataset"); ap.add_argument("--model")
    ap.add_argument("--seed", type=int)
    return ap


if __name__ == "__main__":
    a = build_argparser().parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.sampler is None:
        a.sampler = cfg["tuning"]["sampler"]
    if a.worker:
        run_worker(a, cfg)
    else:
        run_orchestrator(a, cfg)
