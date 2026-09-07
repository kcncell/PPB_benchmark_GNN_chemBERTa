#!/usr/bin/env python
"""Late fusion of GCN + GIN + GAT on the same molecular graph (no ChemBERTa).

Control for ChemBERTa+GNN fusion: three graph encoders matching the standalone
PyG models (3 x 128, dropout 0.25, GAT heads=4), mean-pool, concatenate, MLP
head as in train_multimodal_fusion.py.

Trained with the standalone GNN protocol (100 epochs, patience 25, Adam 1e-3,
weight decay 1e-4) so the control is not under-fit relative to GCN/GIN/GAT.

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE
    conda activate dc
    cd Project_1
    python scripts/train_gnn_triple_fusion.py --dataset ppb --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import deepchem as dc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from shared.data_loaders import DATASET_REGISTRY, DEFAULT_DATA_DIR, DEFAULT_SAVE_DIR  # noqa: E402
from shared.device import get_torch_device  # noqa: E402
from shared.metrics import metrics_to_row, regression_metrics  # noqa: E402
from shared.results_io import append_leaderboard, save_metrics  # noqa: E402
from shared.seed import set_seed  # noqa: E402
from train_pyg_gnn import dc_to_pyg_loader  # noqa: E402

DEFAULT_DATASETS = ["ppb"]


class GraphEncoder(nn.Module):
    """Same graph stack as train_pyg_gnn.MoleculeGNN, minus the regression Linear."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.25,
        arch: str = "gin",
        heads: int = 4,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.out_dim = hidden

        if self.arch == "gat":
            if hidden % heads != 0:
                raise ValueError("hidden must be divisible by GAT heads")
            head_dim = hidden // heads
            for i in range(layers):
                in_c = in_dim if i == 0 else hidden
                concat = i < layers - 1
                if concat:
                    self.convs.append(
                        GATConv(in_c, head_dim, heads=heads, dropout=dropout, concat=True)
                    )
                    bn_dim = heads * head_dim
                else:
                    self.convs.append(
                        GATConv(in_c, hidden, heads=heads, dropout=dropout, concat=False)
                    )
                    bn_dim = hidden
                self.bns.append(nn.BatchNorm1d(bn_dim))
        else:
            dims = [in_dim] + [hidden] * layers
            for i in range(layers):
                if self.arch == "gin":
                    mlp = nn.Sequential(
                        nn.Linear(dims[i], dims[i + 1]),
                        nn.ReLU(),
                        nn.Linear(dims[i + 1], dims[i + 1]),
                    )
                    self.convs.append(GINConv(mlp))
                elif self.arch == "gcn":
                    self.convs.append(GCNConv(dims[i], dims[i + 1]))
                else:
                    raise ValueError(f"Unknown graph encoder: {arch}")
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class TripleGNNFusion(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_tasks: int,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.25,
        fusion_hidden: int = 256,
        head_dropout: float = 0.2,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.gcn = GraphEncoder(in_dim, hidden, layers, dropout, arch="gcn")
        self.gin = GraphEncoder(in_dim, hidden, layers, dropout, arch="gin")
        self.gat = GraphEncoder(in_dim, hidden, layers, dropout, arch="gat", heads=gat_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(fusion_hidden // 2, n_tasks),
        )

    def forward(self, x, edge_index, batch):
        fused = torch.cat(
            [
                self.gcn(x, edge_index, batch),
                self.gin(x, edge_index, batch),
                self.gat(x, edge_index, batch),
            ],
            dim=-1,
        )
        return self.head(fused)


@torch.no_grad()
def evaluate(model, loader, device, n_tasks: int) -> dict:
    model.eval()
    preds, labels, weights = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds.append(out.cpu().numpy())
        y = data.y
        if y.dim() == 1:
            y = y.view(-1, n_tasks)
        labels.append(y.cpu().numpy())
        w = data.w
        if w.dim() == 1:
            w = w.view(-1, n_tasks)
        weights.append(w.cpu().numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(labels)
    w = np.vstack(weights)
    if n_tasks == 1:
        mask = w.ravel() > 0
        return regression_metrics(y_true.ravel()[mask], y_pred.ravel()[mask])
    task_rmses, task_maes, task_r2s = [], [], []
    for t in range(n_tasks):
        mask = w[:, t] > 0
        if mask.sum() < 2:
            continue
        m = regression_metrics(y_true[mask, t], y_pred[mask, t])
        task_rmses.append(m["rmse"])
        task_maes.append(m["mae"])
        task_r2s.append(m["r2"])
    return {
        "rmse": float(np.mean(task_rmses)) if task_rmses else float("nan"),
        "mae": float(np.mean(task_maes)) if task_maes else float("nan"),
        "r2": float(np.mean(task_r2s)) if task_r2s else float("nan"),
        "n_tasks_scored": len(task_r2s),
    }


def load_molgraph(dataset_name: str):
    meta = DATASET_REGISTRY[dataset_name]
    loader_fn = getattr(dc.molnet, meta["loader"])
    save_dir = DEFAULT_SAVE_DIR / dataset_name / "MolGraphConv_scaffold"
    save_dir.mkdir(parents=True, exist_ok=True)
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=False)
    tasks, datasets, _ = loader_fn(
        featurizer=featurizer,
        splitter=meta["default_splitter"],
        transformers=["normalization"],
        reload=True,
        data_dir=str(DEFAULT_DATA_DIR),
        save_dir=str(save_dir),
    )
    return list(tasks), datasets, meta["default_splitter"]


def train_one(
    dataset_name: str,
    seed: int,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden: int = 128,
    num_layers: int = 3,
    dropout: float = 0.25,
    weight_decay: float = 1e-4,
    patience: int = 25,
    fusion_hidden: int = 256,
    gat_heads: int = 4,
):
    set_seed(seed)
    device = get_torch_device()
    tasks, (train_ds, valid_ds, test_ds), splitter = load_molgraph(dataset_name)
    n_tasks = len(tasks)

    train_loader, n_feat, n_train, _, _ = dc_to_pyg_loader(train_ds, batch_size, shuffle=True)
    valid_loader, _, n_valid, _, _ = dc_to_pyg_loader(valid_ds, batch_size, shuffle=False)
    test_loader, _, n_test, _, _ = dc_to_pyg_loader(test_ds, batch_size, shuffle=False)

    model = TripleGNNFusion(
        in_dim=n_feat,
        n_tasks=n_tasks,
        hidden=hidden,
        layers=num_layers,
        dropout=dropout,
        fusion_hidden=fusion_hidden,
        gat_heads=gat_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{dataset_name}] TripleGNN GCN+GIN+GAT seed={seed} "
        f"n={n_train}/{n_valid}/{n_test} params={n_params} device={device} "
        f"graph={num_layers}x{hidden} drop={dropout}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction="none")

    history = []
    best_valid_r2 = -1e9
    best_state = None
    best_epoch = 0
    stale = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            y = data.y
            w = data.w
            if y.dim() == 1:
                y = y.view(out.shape[0], -1)
            if w.dim() == 1:
                w = w.view(out.shape[0], -1)
            loss = criterion(out, y)
            loss = (loss * w).sum() / w.sum().clamp_min(1e-8)
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
            f"[{dataset_name}/triple-gnn] seed={seed} epoch {epoch}/{epochs} "
            f"loss={avg_loss:.4f} train_r2={train_m['r2']:.4f} valid_r2={valid_m['r2']:.4f}"
        )
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        if valid_m["r2"] > best_valid_r2 and np.isfinite(valid_m["r2"]):
            best_valid_r2 = valid_m["r2"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience and epoch > 30:
                print(f"  early stop at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_m = evaluate(model, train_loader, device, n_tasks)
    valid_m = evaluate(model, valid_loader, device, n_tasks)
    test_m = evaluate(model, test_loader, device, n_tasks)
    train_time = time.time() - t0

    model_tag = f"TripleGNN_GCN_GIN_GAT_h{hidden}_e{epochs}"
    payload = {
        "dataset": dataset_name,
        "model": model_tag,
        "arch": "gcn+gin+gat",
        "seed": seed,
        "splitter": splitter,
        "tasks": tasks,
        "n_tasks": n_tasks,
        "n_node_features": n_feat,
        "n_params": n_params,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "lr": lr,
        "hidden": hidden,
        "num_layers": num_layers,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "fusion_hidden": fusion_hidden,
        "gat_heads": gat_heads,
        "device": str(device),
        "train_time_sec": round(train_time, 3),
        "best_valid_r2": best_valid_r2,
        "train": train_m,
        "valid": valid_m,
        "test": test_m,
        "history": history,
        "sizes": {"train": n_train, "valid": n_valid, "test": n_test},
    }
    save_metrics(dataset_name, model_tag, seed, payload)
    append_leaderboard(
        metrics_to_row(
            dataset=dataset_name,
            model=model_tag,
            seed=seed,
            split=splitter,
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
                "hidden": hidden,
                "arch": "gcn+gin+gat",
            },
        )
    )
    print(
        f"[{dataset_name}] TripleGNN seed={seed} best_epoch={best_epoch} "
        f"test R2={test_m['r2']:.4f} RMSE={test_m['rmse']:.4f} ({train_time:.1f}s)"
    )
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ppb")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--fusion-hidden", type=int, default=256)
    p.add_argument("--gat-heads", type=int, default=4)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = DEFAULT_DATASETS if args.dataset == "all" else [args.dataset.lower()]

    for ds in datasets:
        for seed in seeds:
            try:
                train_one(
                    ds,
                    seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    hidden=args.hidden,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    fusion_hidden=args.fusion_hidden,
                    gat_heads=args.gat_heads,
                )
            except Exception as e:
                print(f"[ERROR] {ds} triple-gnn seed={seed}: {e}")
                import traceback

                traceback.print_exc()
            finally:
                gc.collect()
                try:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
