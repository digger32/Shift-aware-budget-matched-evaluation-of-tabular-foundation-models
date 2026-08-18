#!/usr/bin/env python3
"""Per-unit work for the evaluation grid. One unit = (dataset, model, seed); the fitted
model is evaluated on BOTH the ID and (if present) OOD test splits, so shift
degradation is computed within a single unit under one frozen subsample.

Anti-OOM design (following the PyTorch CUDA memory-management notes):
  - the unit runs in its own subprocess (fragmentation cannot accumulate);
  - PYTORCH_ALLOC_CONF=expandable_segments:True is exported by pipeline.sh;
  - prediction is chunked; on torch.OutOfMemoryError the chunk is halved and
    retried down to 256 rows, then the unit fails alone (status in manifest);
  - peak VRAM (allocated and reserved) goes into the unit JSON.

Fairness invariants asserted by the gate:
  - subsample indices depend ONLY on (dataset, seed) — identical for every
    model; their sha256 goes into the JSON (gate check F1);
  - device per model follows configs/grid.yaml (trees on pinned-thread CPU,
    TFMs on GPU) and is recorded (gate check F2);
  - GBDT tuning budget (n_trials, sampler, wall_s) is logged per unit (F3).
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

# Tabular foundation model APIs verified against the pinned releases
# (tabpfn 8.1.0, tabicl 2.1.1). Checkpoints: tabpfn v2 -> HF Prior-Labs/TabPFN-v2-clf;
# tabpfn v2.5 default (real-data finetuned; RealTabPFN, Garg et al. 2025) ->
# HF Prior-Labs/tabpfn_2_5; tabicl v2 -> HF jingang/TabICL
# (tabicl-classifier-v2-20260212.ckpt, the package default).


# ----------------------------------------------------------------- data ----- #
def load_dataset(data_dir: Path, dataset: str):
    z = np.load(data_dir / f"{dataset}.npz")
    ood = ("X_ood" in z.files)
    return (z["X_train"], z["y_train"], z["X_id"], z["y_id"],
            z["X_ood"] if ood else None, z["y_ood"] if ood else None)


def frozen_subsample(dataset: str, seed: int, n: int, cap: int, tag: str):
    """Indices depend only on (dataset, seed, tag) — never on the model."""
    key = int.from_bytes(hashlib.sha256(
        f"{dataset}|{seed}|{tag}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(key)
    idx = np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))
    h = hashlib.sha256(idx.tobytes()).hexdigest()[:16]
    return idx, h


# -------------------------------------------------------------- metrics ----- #
def ece15(y, p1, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p1 >= lo) & (p1 < hi) if hi < 1 else (p1 >= lo) & (p1 <= hi)
        if m.any():
            e += m.sum() / n * abs(y[m].mean() - p1[m].mean())
    return float(e)


def smece(y, p1):
    """Binning-free smooth ECE (Blasiok & Nakkiran) via relplot if installed."""
    try:
        import relplot
        return float(relplot.smECE(p1, y))
    except Exception:
        return None


def compute_metrics(y, proba):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 log_loss, roc_auc_score)
    pred = proba.argmax(axis=1)
    binary = proba.shape[1] == 2
    out = {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "logloss": float(log_loss(y, proba, labels=list(range(proba.shape[1])))),
    }
    if binary:
        p1 = np.clip(proba[:, 1], 1e-7, 1 - 1e-7)
        out["auroc"] = float(roc_auc_score(y, p1))
        out["brier"] = float(np.mean((p1 - y) ** 2))
        out["ece"] = ece15(y, p1)
        out["smece"] = smece(y, p1)
    else:
        out.update({"auroc": None, "brier": None, "ece": None, "smece": None})
    return out


# ------------------------------------------------------------ prediction ---- #
def predict_proba_chunked(model, X, chunk, uses_torch):
    """Chunked predict_proba with an OOM retry ladder for torch models."""
    if not uses_torch:
        return model.predict_proba(X)
    import torch
    while True:
        try:
            parts = [model.predict_proba(X[i:i + chunk])
                     for i in range(0, len(X), chunk)]
            return np.vstack(parts)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= 256:
                raise
            chunk //= 2
            print(f"[worker] OOM — retrying with chunk={chunk}", flush=True)


# ---------------------------------------------------------------- models ---- #
GBDT_SPACES = {
    "xgb": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 100, 1000),
        max_depth=t.suggest_int("max_depth", 3, 10),
        learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=t.suggest_float("min_child_weight", 1e-2, 10, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True)),
    "lgbm": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 100, 1000),
        num_leaves=t.suggest_int("num_leaves", 15, 255),
        learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples=t.suggest_int("min_child_samples", 5, 100),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True)),
    "catb": lambda t: dict(
        iterations=t.suggest_int("iterations", 100, 1000),
        depth=t.suggest_int("depth", 3, 10),
        learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1e-1, 10, log=True),
        bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 1.0)),
}


def make_gbdt(model, params, seed, threads):
    """CPU-only, with threads pinned: at these sample sizes GPU tree building
    is markedly slower than CPU, and unpinned OpenMP is slower still."""
    if model == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, n_jobs=threads, random_state=seed,
                             tree_method="hist", eval_metric="logloss")
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, n_jobs=threads, random_state=seed,
                              verbosity=-1)
    if model == "catb":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, thread_count=threads,
                                  random_seed=seed, verbose=False)
    raise ValueError(model)


def tune_gbdt(model, Xtr, ytr, seed, tune_cfg, sampler_name):
    """Disclosed-budget HPO on train only (stratified CV, logloss objective).
    Returns final fitted model, tuning log, and the budget curve param sets."""
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    threads, n_trials = tune_cfg["threads"], tune_cfg["n_trials"]
    cv = StratifiedKFold(tune_cfg["cv_folds"], shuffle=True, random_state=seed)

    def objective(trial):
        params = GBDT_SPACES[model](trial)
        est = make_gbdt(model, params, seed, threads)
        return -cross_val_score(est, Xtr, ytr, cv=cv,
                                scoring="neg_log_loss", n_jobs=1).mean()

    sampler = (optuna.samplers.TPESampler(seed=seed) if sampler_name == "tpe"
               else optuna.samplers.RandomSampler(seed=seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    tune_wall = time.time() - t0

    # best-so-far params at each budget checkpoint (0 = library defaults)
    trials = study.trials
    curve_params = {}
    for k in tune_cfg["budget_checkpoints"]:
        if k == 0:
            curve_params[0] = {}
        elif k <= len(trials):
            best = min(trials[:k], key=lambda t: t.value)
            curve_params[k] = best.params
    log = {"n_trials": n_trials, "sampler": sampler_name,
           "cv_folds": tune_cfg["cv_folds"], "wall_s": round(tune_wall, 1),
           "cv_best_logloss": float(study.best_value),
           "best_params": study.best_params}
    return curve_params, log


def make_simple(model, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=seed))
    if model == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier((256, 256), early_stopping=True,
                                           max_iter=300, random_state=seed))
    raise ValueError(model)


def make_tfm(model, device, models_dir, seed):
    """Verified factories (see module note for checkpoint provenance)."""
    if model in ("tabpfn2", "tabpfn25"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        version = ModelVersion.V2 if model == "tabpfn2" else ModelVersion.V2_5
        return TabPFNClassifier.create_default_for_version(
            version, device=device, random_state=seed)
    if model == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=device, random_state=seed)
    raise ValueError(model)


# ------------------------------------------------------------------ unit ---- #
def run_unit(dataset, model, seed, cfg, out_path: Path, sampler=None,
             data_dir=None, preds_dir=None):
    t0 = time.time()
    mcfg = cfg["models"][model]
    device, kind = mcfg["device"], mcfg["kind"]
    data_dir = Path(data_dir or cfg["runtime"]["data_dir"])
    registry = json.loads((data_dir / "registry.json").read_text())
    suite = registry["datasets"][dataset]["suite"]

    Xtr, ytr, Xid, yid, Xood, yood = load_dataset(data_dir, dataset)
    caps = cfg["subsample"]
    itr, htr = frozen_subsample(dataset, seed, len(ytr), caps["train_cap"], "train")
    iid, hid = frozen_subsample(dataset, seed, len(yid), caps["test_cap"], "id")
    Xtr, ytr, Xid, yid = Xtr[itr], ytr[itr], Xid[iid], yid[iid]
    hood = None
    if Xood is not None:
        iod, hood = frozen_subsample(dataset, seed, len(yood), caps["test_cap"], "ood")
        Xood, yood = Xood[iod], yood[iod]

    uses_torch = (kind == "tfm")
    tuning = None
    budget_curve = []

    if kind == "gbdt":
        sampler = sampler or cfg["tuning"]["sampler"]
        curve_params, tuning = tune_gbdt(model, Xtr, ytr, seed,
                                         cfg["tuning"], sampler)
        for k in sorted(curve_params):
            est = make_gbdt(model, curve_params[k], seed, cfg["tuning"]["threads"])
            est.fit(Xtr, ytr)
            row = {"budget": k,
                   "id": compute_metrics(yid, est.predict_proba(Xid))}
            if Xood is not None:
                row["ood"] = compute_metrics(yood, est.predict_proba(Xood))
            budget_curve.append(row)
            if k == max(curve_params):
                clf = est                      # final model = full-budget best
    elif kind == "tfm":
        clf = make_tfm(model, device, cfg["runtime"]["models_dir"], seed)
        clf.fit(Xtr, ytr)
    else:
        clf = make_simple(model, seed)
        clf.fit(Xtr, ytr)

    chunk = cfg["runtime"]["predict_chunk"]
    proba_id = predict_proba_chunked(clf, Xid, chunk, uses_torch)
    m_id = compute_metrics(yid, proba_id)
    m_ood, proba_ood = None, None
    if Xood is not None:
        proba_ood = predict_proba_chunked(clf, Xood, chunk, uses_torch)
        m_ood = compute_metrics(yood, proba_ood)

    peak = {}
    if uses_torch:
        import torch
        if torch.cuda.is_available():
            peak = {"peak_vram_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                    "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}

    # flat metrics block: *_ood suffixed; keys 'ece' and 'optimism_gap' feed the
    # gate's generic C1/D1 checks directly.
    metrics = {k: v for k, v in m_id.items()}
    if m_ood:
        metrics.update({f"{k}_ood": v for k, v in m_ood.items()})
        if m_id["acc"] is not None:
            metrics["acc_degradation"] = m_id["acc"] - m_ood["acc"]
    if tuning:
        metrics["optimism_gap"] = float(m_id["logloss"] - tuning["cv_best_logloss"])

    if preds_dir:
        Path(preds_dir).mkdir(parents=True, exist_ok=True)
        arrays = {"proba_id": proba_id.astype(np.float16), "y_id": yid}
        if proba_ood is not None:
            arrays.update({"proba_ood": proba_ood.astype(np.float16), "y_ood": yood})
        np.savez_compressed(Path(preds_dir) / f"{dataset}__{model}__seed{seed}.npz",
                            **arrays)

    import importlib.metadata as im
    versions = {}
    for pkg in ["numpy", "scikit-learn", "xgboost", "lightgbm", "catboost",
                "optuna", "tabpfn", "tabicl", "torch"]:
        try:
            versions[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            pass

    result = {
        "dataset": dataset, "suite": suite, "model": model, "seed": seed,
        "device": device, "kind": kind,
        "subsample": {"train_hash": htr, "id_hash": hid, "ood_hash": hood,
                      "n_train": int(len(ytr)), "n_id": int(len(yid)),
                      "n_ood": int(len(yood)) if yood is not None else 0},
        "metrics": metrics, "tuning": tuning, "budget_curve": budget_curve,
        "wall_s": round(time.time() - t0, 1), **peak, "versions": versions,
    }
    out_path.write_text(json.dumps(result, indent=2))
    return result
