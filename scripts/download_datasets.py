#!/usr/bin/env python
"""Download / cache Project 1 MoleculeNet datasets via DeepChem loaders.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets ppb,hppb,hopv
    python scripts/download_datasets.py --datasets clearance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_loaders import load_dataset, list_datasets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Cache Project 1 MolNet datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ppb,hppb,hopv,clearance",
        help="Comma-separated dataset names or 'all'",
    )
    parser.add_argument(
        "--featurizer",
        type=str,
        default="ECFP",
        help="Featurizer used for cache (ECFP default for RF baselines)",
    )
    args = parser.parse_args()

    if args.datasets.lower() == "all":
        names = list_datasets()
    else:
        names = [n.strip().lower() for n in args.datasets.split(",") if n.strip()]

    summary = []
    for name in names:
        print("\n" + "=" * 60)
        print(f"Downloading / caching: {name}")
        print("=" * 60)
        try:
            loaded = load_dataset(name, featurizer=args.featurizer, reload=True)
            summary.append(
                {
                    "dataset": name,
                    "status": "ok",
                    "tasks": len(loaded.tasks),
                    "splitter": loaded.splitter,
                    "train": len(loaded.train),
                    "valid": len(loaded.valid),
                    "test": len(loaded.test),
                }
            )
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            summary.append({"dataset": name, "status": f"fail: {exc}"})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
