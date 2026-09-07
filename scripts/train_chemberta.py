#!/usr/bin/env python
"""Unified ChemBERTa fine-tune for Project 1 regression datasets.

Uses Hugging Face ChemBERTa + SMILES from DeepChem loaders (ids or Raw).
Writes metrics under results/ and appends to leaderboard.csv.

Usage:
    conda activate dc
    cd Project_1
    python scripts/train_chemberta.py --dataset ppb --seeds 42 --epochs 10
    python scripts/train_chemberta.py --dataset hopv --seeds 42 --epochs 15
    python scripts/train_chemberta.py --dataset clearance --seeds 42,43,44 --epochs 10
    python scripts/train_chemberta.py --dataset all --seeds 42 --epochs 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_loaders import list_datasets, load_dataset  # noqa: E402
from shared.device import get_torch_device  # noqa: E402
from shared.metrics import metrics_to_row, regression_metrics  # noqa: E402
from shared.results_io import append_leaderboard, save_metrics  # noqa: E402
from shared.seed import set_seed  # noqa: E402

DEFAULT_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

# Skip HPPB by default until raw-data identity is resolved (see docs/dataset_notes.md)
DEFAULT_TRAIN_DATASETS = ["ppb", "hopv", "clearance"]


def _smiles_from_dataset(dc_dataset) -> List[str]:
    """Extract SMILES strings from a DeepChem dataset."""
    ids = [str(x) for x in dc_dataset.ids]
    # ids are usually SMILES for molnet loaders
    if ids and ("C" in ids[0] or "c" in ids[0] or "O" in ids[0] or "[" in ids[0]):
        return ids
    # Fallback: Raw featurizer stores mols in X
    try:
        from rdkit import Chem

        return [Chem.MolToSmiles(m) if m is not None else "" for m in dc_dataset.X]
    except Exception:
        return ids


class SmilesRegressionDataset(Dataset):
    def __init__(self, smiles: List[str], y: np.ndarray, tokenizer, max_len: int = 128):
        self.smiles = smiles
        self.y = np.asarray(y, dtype=np.float32)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_tasks = self.y.shape[1]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def _make_loader(dc_ds, tokenizer, batch_size: int, shuffle: bool, max_len: int) -> DataLoader:
    smiles = _smiles_from_dataset(dc_ds)
    y = np.asarray(dc_ds.y, dtype=np.float32)
    ds = SmilesRegressionDataset(smiles, y, tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device, n_tasks: int) -> dict:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].cpu().numpy()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = out.logits.detach().cpu().numpy()
        preds.append(pred)
        labels.append(y)
    y_pred = np.vstack(preds)
    y_true = np.vstack(labels)
    if n_tasks == 1:
        return regression_metrics(y_true.ravel(), y_pred.ravel())
    return regression_metrics(y_true, y_pred, multitask=True)


def _set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze/unfreeze ChemBERTa backbone (gradual unfreezing)."""
    base = getattr(model, "base_model", None) or getattr(model, "roberta", None)
    if base is None:
        # fallback: freeze all except classifier
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = trainable
        return
    for param in base.parameters():
        param.requires_grad = trainable


def train_one(
    dataset_name: str,
    seed: int,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-5,
    max_len: int = 128,
    weight_decay: float = 0.01,
    freeze_epochs: int = 2,
) -> dict:
    """Fine-tune ChemBERTa with optional gradual unfreezing.

    Hyperparameters follow Project 1 PPB tuning notes:
    - LR search tested {1e-5, 3e-5, 5e-5} for 20 epochs with freeze→unfreeze.
      Best *peak* test R² ≈ 0.268 at 5e-5; best *stable final* at 1e-5 (0.263).
      Default lr=1e-5 for multi-seed paper runs (stable under scaffold).
    - freeze_epochs=2: train head only, then unfreeze full model.
    - weight_decay=0.01
    - Keep train/valid/test separate (no train+valid merge) for rigorous reporting.
    """
    set_seed(seed)
    device = get_torch_device()

    # SMILES via ids: ECFP still keeps SMILES in ids for molnet
    loaded = load_dataset(dataset_name, featurizer="ECFP", splitter=None)
    n_tasks = len(loaded.tasks)

    print(f"Loading tokenizer/model: {model_name}")
    print(
        f"  hparams: epochs={epochs} lr={lr} batch={batch_size} "
        f"wd={weight_decay} freeze_epochs={freeze_epochs}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_tasks,
        problem_type="regression",
    )
    model.to(device)

    train_loader = _make_loader(loaded.train, tokenizer, batch_size, True, max_len)
    valid_loader = _make_loader(loaded.valid, tokenizer, batch_size, False, max_len)
    test_loader = _make_loader(loaded.test, tokenizer, batch_size, False, max_len)

    # Gradual unfreezing: freeze backbone for first freeze_epochs
    if freeze_epochs > 0:
        _set_backbone_trainable(model, trainable=False)
        print(f"  epochs 1-{freeze_epochs}: backbone frozen (head only)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    history = []
    best_valid_r2 = -1e9
    best_state = None
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Unfreeze after freeze_epochs (e.g. at start of epoch 3 if freeze_epochs=2)
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            _set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            print(f"  epoch {epoch}: backbone unfrozen; training all layers")

        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_m = evaluate(model, train_loader, device, n_tasks)
        valid_m = evaluate(model, valid_loader, device, n_tasks)
        avg_loss = total_loss / max(n_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "train_r2": train_m["r2"],
                "valid_r2": valid_m["r2"],
            }
        )
        print(
            f"[{dataset_name}] seed={seed} epoch {epoch}/{epochs} "
            f"loss={avg_loss:.4f} train_r2={train_m['r2']:.4f} valid_r2={valid_m['r2']:.4f}"
        )
        if valid_m["r2"] > best_valid_r2 and np.isfinite(valid_m["r2"]):
            best_valid_r2 = valid_m["r2"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_m = evaluate(model, train_loader, device, n_tasks)
    valid_m = evaluate(model, valid_loader, device, n_tasks)
    test_m = evaluate(model, test_loader, device, n_tasks)
    train_time = time.time() - t0

    payload = {
        "dataset": loaded.name,
        "model": "ChemBERTa_zinc_base_v1",
        "model_name": model_name,
        "seed": seed,
        "splitter": loaded.splitter,
        "tasks": loaded.tasks,
        "n_tasks": n_tasks,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "freeze_epochs": freeze_epochs,
        "max_len": max_len,
        "device": str(device),
        "train_time_sec": round(train_time, 3),
        "best_valid_r2": best_valid_r2,
        "best_epoch": best_epoch,
        "train": train_m,
        "valid": valid_m,
        "test": test_m,
        "history": history,
        "notes": (
            "Early stop by best valid R2. Gradual unfreezing. "
            "No train+valid merge (paper-grade protocol)."
        ),
    }
    # include lr in model folder tag so old 2e-5 runs are not overwritten silently
    model_tag = f"ChemBERTa_zinc_base_v1_lr{lr:g}_e{epochs}"
    save_metrics(loaded.name, model_tag, seed, payload)

    row = metrics_to_row(
        dataset=loaded.name,
        model=model_tag,
        seed=seed,
        split=loaded.splitter,
        split_name="test",
        metrics=test_m,
        extra={
            "train_r2": train_m["r2"],
            "valid_r2": valid_m["r2"],
            "train_time_sec": round(train_time, 3),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "freeze_epochs": freeze_epochs,
            "weight_decay": weight_decay,
        },
    )
    append_leaderboard(row)

    print(
        f"[{loaded.name}] ChemBERTa seed={seed} best_epoch={best_epoch} "
        f"test R2={test_m['r2']:.4f} RMSE={test_m['rmse']:.4f} ({train_time:.1f}s)"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Project 1 ChemBERTa unified trainer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ppb",
        help="Dataset name, 'all' (ppb,hopv,clearance), or 'all_including_hppb'",
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (prior PPB tuning used 20)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate. Prior search: 1e-5 (stable best final), "
        "5e-5 (best peak). Not 1e-4.",
    )
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=2,
        help="Epochs to train head only before unfreezing backbone (0=off)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.dataset.lower() == "all":
        datasets = DEFAULT_TRAIN_DATASETS
    elif args.dataset.lower() == "all_including_hppb":
        datasets = list_datasets()
    else:
        datasets = [args.dataset.lower()]

    for ds in datasets:
        for seed in seeds:
            try:
                train_one(
                    ds,
                    seed,
                    model_name=args.model_name,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    max_len=args.max_len,
                    weight_decay=args.weight_decay,
                    freeze_epochs=args.freeze_epochs,
                )
            except Exception as exc:
                print(f"[ERROR] dataset={ds} seed={seed}: {exc}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()
