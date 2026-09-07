"""Save metrics JSON and maintain a global leaderboard CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
LEADERBOARD_PATH = RESULTS_ROOT / "leaderboard.csv"


def save_metrics(
    dataset: str,
    model: str,
    seed: int,
    metrics: Dict[str, Any],
    results_root: Optional[Path] = None,
) -> Path:
    """Write metrics.json under results/{dataset}/{model}/seed{N}/."""
    root = Path(results_root) if results_root else RESULTS_ROOT
    out_dir = root / dataset / model / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def append_leaderboard(
    row: Dict[str, Any],
    leaderboard_path: Optional[Path] = None,
) -> Path:
    """Append one result row to the project leaderboard CSV."""
    path = Path(leaderboard_path) if leaderboard_path else LEADERBOARD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    return path
