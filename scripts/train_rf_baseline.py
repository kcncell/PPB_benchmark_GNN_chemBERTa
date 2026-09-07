#!/usr/bin/env python
"""Multi-seed Random Forest + ECFP baseline for Project 1 datasets.

Usage (from Project_1 root, with conda env `dc` or `deepchem`):

    python scripts/train_rf_baseline.py --dataset ppb --seeds 42,43,44
    python scripts/train_rf_baseline.py --dataset hopv --seeds 42
    python scripts/train_rf_baseline.py --dataset all --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Allow importing shared/ when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_loaders import load_dataset, list_datasets  # noqa: E402
from shared.metrics import regression_metrics, metrics_to_row  # noqa: E402
from shared.results_io import append_leaderboard, save_metrics  # noqa: E402
from shared.seed import set_seed  # noqa: E402


def _predict_dataset(model, dataset) -> np.ndarray:
    """Sklearn RF on DeepChem dataset X."""
    X = np.asarray(dataset.X)
    if X.ndim == 1:
        # rare: object arrays
        X = np.stack(X)
    return model.predict(X)


def _y(dataset) -> np.ndarray:
    return np.asarray(dataset.y, dtype=float)


def train_one(
    dataset_name: str,
    seed: int,
    n_estimators: int = 200,
    splitter: str | None = None,
) -> dict:
    set_seed(seed)
    loaded = load_dataset(dataset_name, featurizer="ECFP", splitter=splitter)
    model_name = (
        "RandomForest_ECFP_random" if loaded.splitter == "random" else "RandomForest_ECFP"
    )

    X_train = np.asarray(loaded.train.X)
    y_train = _y(loaded.train)
    if y_train.ndim > 1 and y_train.shape[1] == 1:
        y_train_fit = y_train.ravel()
    else:
        y_train_fit = y_train

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train_fit)
    train_time = time.time() - t0

    rows = {}
    for split_name, ds in [
        ("train", loaded.train),
        ("valid", loaded.valid),
        ("test", loaded.test),
    ]:
        pred = _predict_dataset(model, ds)
        y_true = _y(ds)
        # Align shapes for multitask
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if y_true.shape[1] == 1:
            m = regression_metrics(y_true.ravel(), pred.ravel())
        else:
            m = regression_metrics(y_true, pred, multitask=True)
        rows[split_name] = m

    payload = {
        "dataset": loaded.name,
        "model": model_name,
        "seed": seed,
        "splitter": loaded.splitter,
        "featurizer": loaded.featurizer,
        "tasks": loaded.tasks,
        "n_estimators": n_estimators,
        "train_time_sec": round(train_time, 3),
        "train": rows["train"],
        "valid": rows["valid"],
        "test": rows["test"],
    }
    save_metrics(loaded.name, model_name, seed, payload)

    leaderboard_row = metrics_to_row(
        dataset=loaded.name,
        model=model_name,
        seed=seed,
        split=loaded.splitter,
        split_name="test",
        metrics=rows["test"],
        extra={
            "train_r2": rows["train"]["r2"],
            "valid_r2": rows["valid"]["r2"],
            "train_time_sec": round(train_time, 3),
            "n_estimators": n_estimators,
        },
    )
    append_leaderboard(leaderboard_row)

    print(
        f"[{loaded.name}] RF seed={seed} splitter={loaded.splitter} "
        f"test R2={rows['test']['r2']:.4f} RMSE={rows['test']['rmse']:.4f} "
        f"({train_time:.1f}s)"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Project 1 RF + ECFP multi-seed baseline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ppb",
        help=f"Dataset name or 'all'. Options: {list_datasets()}",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--splitter",
        type=str,
        default=None,
        help="DeepChem splitter (default: dataset registry, scaffold for PPB). "
        "Use 'random' for the RF random-split control.",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.dataset.lower() == "all":
        datasets = list_datasets()
    else:
        datasets = [args.dataset.lower()]

    for ds in datasets:
        for seed in seeds:
            try:
                train_one(
                    ds,
                    seed,
                    n_estimators=args.n_estimators,
                    splitter=args.splitter,
                )
            except Exception as exc:
                print(f"[ERROR] dataset={ds} seed={seed}: {exc}")


if __name__ == "__main__":
    main()
