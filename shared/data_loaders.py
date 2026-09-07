"""Unified DeepChem MoleculeNet loaders for Project 1 datasets.

Datasets: PPB, HPPB, HOPV, Clearance (optional pilot).
Default split policy: scaffold for ADME sets; scaffold for HOPV when possible,
with documented fallback to random if the loader rejects scaffold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import deepchem as dc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "data" / "featurized"

# Registry of Project 1 datasets
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ppb": {
        "loader": "load_ppb",
        "default_splitter": "scaffold",
        "task_type": "regression",
        "domain": "adme",
    },
    "hppb": {
        "loader": "load_hppb",
        "default_splitter": "scaffold",
        "task_type": "regression",
        "domain": "adme",
    },
    "hopv": {
        "loader": "load_hopv",
        "default_splitter": "scaffold",
        "task_type": "regression_multitask",
        "domain": "materials_opv",
    },
    "clearance": {
        "loader": "load_clearance",
        "default_splitter": "scaffold",
        "task_type": "regression",
        "domain": "adme",
    },
}


@dataclass
class LoadedDataset:
    name: str
    tasks: List[str]
    train: Any
    valid: Any
    test: Any
    transformers: list
    featurizer: str
    splitter: str
    multitask: bool


def _ensure_dirs(data_dir: Path, save_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    # DeepChem respects DEEPCHEM_DATA_DIR for caching
    os.environ.setdefault("DEEPCHEM_DATA_DIR", str(data_dir))


def load_dataset(
    name: str,
    featurizer: str = "ECFP",
    splitter: Optional[str] = None,
    transformers: Optional[list] = None,
    data_dir: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    reload: bool = True,
) -> LoadedDataset:
    """Load a Project 1 MoleculeNet dataset with standardized options.

    Parameters
    ----------
    name : str
        One of: ppb, hppb, hopv, clearance
    featurizer : str
        DeepChem featurizer shortcut, e.g. 'ECFP', 'GraphConv', or a custom object.
    splitter : str, optional
        Defaults from DATASET_REGISTRY (usually scaffold).
    """
    key = name.lower().strip()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_REGISTRY)}")

    meta = DATASET_REGISTRY[key]
    splitter = splitter or meta["default_splitter"]
    if transformers is None:
        transformers = ["normalization"]

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    save_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR / key / f"{featurizer}_{splitter}"
    _ensure_dirs(data_dir, save_dir)

    loader_fn = getattr(dc.molnet, meta["loader"])

    try:
        tasks, datasets, trans = loader_fn(
            featurizer=featurizer,
            splitter=splitter,
            transformers=transformers,
            reload=reload,
            data_dir=str(data_dir),
            save_dir=str(save_dir),
        )
    except Exception as exc:
        # Some loaders historically prefer random; fall back once with warning.
        if splitter != "random":
            print(
                f"[data_loaders] {key}: splitter='{splitter}' failed ({exc}). "
                f"Falling back to splitter='random'."
            )
            save_dir = DEFAULT_SAVE_DIR / key / f"{featurizer}_random"
            _ensure_dirs(data_dir, save_dir)
            tasks, datasets, trans = loader_fn(
                featurizer=featurizer,
                splitter="random",
                transformers=transformers,
                reload=reload,
                data_dir=str(data_dir),
                save_dir=str(save_dir),
            )
            splitter = "random"
        else:
            raise

    train, valid, test = datasets
    multitask = len(tasks) > 1 or meta["task_type"] == "regression_multitask"

    print(f"[data_loaders] Loaded {key}")
    print(f"  tasks ({len(tasks)}): {tasks}")
    print(f"  featurizer={featurizer} splitter={splitter}")
    print(f"  train={_size(train)} valid={_size(valid)} test={_size(test)}")

    return LoadedDataset(
        name=key,
        tasks=list(tasks),
        train=train,
        valid=valid,
        test=test,
        transformers=trans,
        featurizer=str(featurizer),
        splitter=splitter,
        multitask=multitask,
    )


def _size(ds) -> str:
    try:
        return str(len(ds))
    except Exception:
        try:
            return str(ds.X.shape[0])
        except Exception:
            return "?"


def list_datasets() -> List[str]:
    return list(DATASET_REGISTRY.keys())
