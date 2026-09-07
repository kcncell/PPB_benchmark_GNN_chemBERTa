"""Regression metrics used across Project 1 datasets."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _to_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim > 1 and y.shape[1] == 1:
        return y.ravel()
    return y


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multitask: bool = False,
    sample_weight: Optional[np.ndarray] = None,
    weight_threshold: float = 0.0,
) -> Dict[str, float]:
    """Compute RMSE, MAE, R2 (and multitask means if y is 2D with >1 task).

    sample_weight (optional): DeepChem-style weights. For multitask y of shape
    (N, T), task t is scored only where weight[:, t] > weight_threshold.
    Use this for HOPV missing labels (w=0).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1 or y_true.shape[1] == 1:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).ravel()
            mask = (w > weight_threshold) & np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() < 2:
                return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            return _single_task_metrics(yt[mask], yp[mask])
        return _single_task_metrics(yt, yp)

    task_rmses, task_maes, task_r2s = [], [], []
    per_task: Dict[str, Dict[str, float]] = {}
    n_tasks = y_true.shape[1]
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    for t in range(n_tasks):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if w is not None:
            if w.ndim == 1:
                mask = mask & (w > weight_threshold)
            else:
                mask = mask & (w[:, t] > weight_threshold)
        if mask.sum() < 2:
            continue
        m = _single_task_metrics(yt[mask], yp[mask])
        task_rmses.append(m["rmse"])
        task_maes.append(m["mae"])
        task_r2s.append(m["r2"])
        per_task[str(t)] = {**m, "n": int(mask.sum())}

    if not task_rmses:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "n_tasks_scored": 0,
        }

    out: Dict[str, Any] = {
        "rmse": float(np.mean(task_rmses)),
        "mae": float(np.mean(task_maes)),
        "r2": float(np.mean(task_r2s)),
        "n_tasks_scored": int(len(task_r2s)),
        "per_task": per_task,
    }
    if multitask:
        out["rmse_per_task_mean"] = out["rmse"]
        out["r2_per_task_mean"] = out["r2"]
    return out


def _single_task_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def metrics_to_row(
    *,
    dataset: str,
    model: str,
    seed: int,
    split: str,
    split_name: str,
    metrics: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flatten metrics into a leaderboard row."""
    row: Dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "split": split,
        "split_name": split_name,
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
    }
    if extra:
        row.update(extra)
    return row
