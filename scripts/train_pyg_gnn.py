#!/usr/bin/env python
"""Unified PyTorch Geometric GIN / GCN trainer for Project 1.

Uses DeepChem MolGraphConvFeaturizer + scaffold splits, trains native PyG models.

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE   # needed on some macOS OpenMP setups
    conda activate dc
    cd Project_1
    python scripts/train_pyg_gnn.py --arch gin --dataset all --seeds 42,43,44
    python scripts/train_pyg_gnn.py --arch gcn --dataset all --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# OpenMP conflict workaround on macOS (conda torch + other libs)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_loaders import list_datasets, load_dataset  # noqa: E402
from shared.device import get_torch_device  # noqa: E402
from shared.metrics import metrics_to_row, regression_metrics  # noqa: E402
from shared.results_io import append_leaderboard, save_metrics  # noqa: E402
from shared.seed import set_seed  # noqa: E402

DEFAULT_DATASETS = ["ppb", "hopv", "clearance"]


def _filter_valid_graphs(dc_dataset) -> Tuple[list, np.ndarray, np.ndarray]:
    """Drop failed featurizations (None / empty graphs)."""
    graphs, ys, ws = [], [], []
    X, y, w = dc_dataset.X, np.asarray(dc_dataset.y), np.asarray(dc_dataset.w)
    for i in range(len(X)):
        g = X[i]
        if g is None:
            continue
        if not hasattr(g, "node_features") or g.node_features is None:
            continue
        if len(g.node_features) == 0:
            continue
        graphs.append(g)
        ys.append(y[i])
        ws.append(w[i])
    return graphs, np.asarray(ys, dtype=np.float32), np.asarray(ws, dtype=np.float32)


def dc_to_pyg_loader(
    dc_dataset,
    batch_size: int = 32,
    shuffle: bool = False,
) -> Tuple[DataLoader, int, int, np.ndarray, np.ndarray]:
    """Convert DeepChem MolGraphConv dataset → PyG DataLoader.

    Returns loader, n_node_features, n_kept, y_array, w_array (aligned to loader order).
    """
    graphs, y, w = _filter_valid_graphs(dc_dataset)
    if not graphs:
        raise RuntimeError("No valid graphs after featurization filtering")

    data_list = []
    for i, g in enumerate(graphs):
        x = torch.tensor(g.node_features, dtype=torch.float32)
        edge_index = torch.tensor(g.edge_index, dtype=torch.long)
        # Ensure edge_index shape [2, E]
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        elif edge_index.dim() == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()
        yi = torch.tensor(y[i], dtype=torch.float32).view(1, -1)
        wi = torch.tensor(w[i], dtype=torch.float32).view(1, -1)
        data_list.append(Data(x=x, edge_index=edge_index, y=yi, w=wi))

    n_feat = int(data_list[0].x.shape[1])
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    return loader, n_feat, len(data_list), y, w


class MoleculeGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        arch: str = "gin",
        num_layers: int = 3,
        dropout: float = 0.25,
        heads: int = 4,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.dropout = dropout
        self.heads = heads
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        def make_gin(in_c, out_c):
            nn = torch.nn.Sequential(
                torch.nn.Linear(in_c, out_c),
                torch.nn.ReLU(),
                torch.nn.Linear(out_c, out_c),
            )
            return GINConv(nn)

        if self.arch == "gat":
            # Multi-head GAT: intermediate layers concat heads; last uses mean.
            # hidden_channels is the total embedding dim after each layer.
            assert hidden_channels % heads == 0, "hidden must be divisible by heads"
            head_dim = hidden_channels // heads
            for i in range(num_layers):
                in_c = in_channels if i == 0 else hidden_channels
                concat = i < num_layers - 1
                out_per_head = head_dim if concat else hidden_channels
                # For last layer with concat=False, out channels = heads * out_per_head
                # if we set out_channels=hidden_channels and concat=False: output is heads*out?
                # PyG GATConv: if concat=False, out = out_channels (averaged over heads)
                # if concat=True, out = heads * out_channels
                if concat:
                    self.convs.append(
                        GATConv(in_c, head_dim, heads=heads, dropout=dropout, concat=True)
                    )
                    bn_dim = heads * head_dim  # == hidden_channels
                else:
                    self.convs.append(
                        GATConv(in_c, hidden_channels, heads=heads, dropout=dropout, concat=False)
                    )
                    bn_dim = hidden_channels
                self.batch_norms.append(torch.nn.BatchNorm1d(bn_dim))
        else:
            channels = [in_channels] + [hidden_channels] * num_layers
            for i in range(num_layers):
                if self.arch == "gin":
                    self.convs.append(make_gin(channels[i], channels[i + 1]))
                elif self.arch == "gcn":
                    self.convs.append(GCNConv(channels[i], channels[i + 1]))
                else:
                    raise ValueError(f"Unknown arch: {arch}")
                self.batch_norms.append(torch.nn.BatchNorm1d(channels[i + 1]))

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


@torch.no_grad()
def evaluate(model, loader, device, n_tasks: int) -> dict:
    model.eval()
    preds, labels, weights = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds.append(out.cpu().numpy())
        # data.y is [B, T] after batching
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

    # Multitask: per-task then mean, using weights
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


def train_one(
    dataset_name: str,
    seed: int,
    arch: str = "gin",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden: int = 128,
    num_layers: int = 3,
    dropout: float = 0.25,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> dict:
    set_seed(seed)
    device = get_torch_device()

    # Always use MolGraphConv → PyG-compatible GraphData (not classic ConvMol)
    import deepchem as dc
    from shared.data_loaders import DATASET_REGISTRY, DEFAULT_DATA_DIR, DEFAULT_SAVE_DIR

    meta = DATASET_REGISTRY[dataset_name]
    loader_fn = getattr(dc.molnet, meta["loader"])
    save_dir = DEFAULT_SAVE_DIR / dataset_name / "MolGraphConv_scaffold"
    save_dir.mkdir(parents=True, exist_ok=True)
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=False)
    try:
        tasks, datasets, _trans = loader_fn(
            featurizer=featurizer,
            splitter=meta["default_splitter"],
            transformers=["normalization"],
            reload=True,
            data_dir=str(DEFAULT_DATA_DIR),
            save_dir=str(save_dir),
        )
        splitter = meta["default_splitter"]
    except Exception as exc:
        print(f"[warn] scaffold failed ({exc}); trying random")
        save_dir = DEFAULT_SAVE_DIR / dataset_name / "MolGraphConv_random"
        save_dir.mkdir(parents=True, exist_ok=True)
        tasks, datasets, _trans = loader_fn(
            featurizer=featurizer,
            splitter="random",
            transformers=["normalization"],
            reload=True,
            data_dir=str(DEFAULT_DATA_DIR),
            save_dir=str(save_dir),
        )
        splitter = "random"

    train_ds, valid_ds, test_ds = datasets
    n_tasks = len(tasks)
    task_names = list(tasks)

    train_loader, n_feat, n_train, _, _ = dc_to_pyg_loader(train_ds, batch_size, shuffle=True)
    valid_loader, _, n_valid, _, _ = dc_to_pyg_loader(valid_ds, batch_size, shuffle=False)
    test_loader, _, n_test, _, _ = dc_to_pyg_loader(test_ds, batch_size, shuffle=False)

    print(
        f"[{dataset_name}] {arch.upper()} seed={seed} "
        f"n_feat={n_feat} tasks={n_tasks} sizes={n_train}/{n_valid}/{n_test} device={device}"
    )

    model = MoleculeGNN(
        in_channels=n_feat,
        hidden_channels=hidden,
        out_channels=n_tasks,
        arch=arch,
        num_layers=num_layers,
        dropout=dropout,
        heads=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss(reduction="none")

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
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{dataset_name}/{arch}] seed={seed} epoch {epoch}/{epochs} "
                f"loss={avg_loss:.4f} train_r2={train_m['r2']:.4f} valid_r2={valid_m['r2']:.4f}"
            )

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

    model_tag = f"PyG_{arch.upper()}_h{hidden}_e{epochs}"
    payload = {
        "dataset": dataset_name,
        "model": model_tag,
        "arch": arch,
        "seed": seed,
        "splitter": splitter,
        "tasks": task_names,
        "n_tasks": n_tasks,
        "n_node_features": n_feat,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "lr": lr,
        "hidden": hidden,
        "num_layers": num_layers,
        "dropout": dropout,
        "weight_decay": weight_decay,
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

    row = metrics_to_row(
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
            "arch": arch,
        },
    )
    append_leaderboard(row)

    print(
        f"[{dataset_name}] {arch.upper()} seed={seed} best_epoch={best_epoch} "
        f"test R2={test_m['r2']:.4f} RMSE={test_m['rmse']:.4f} ({train_time:.1f}s)"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Project 1 PyG GIN/GCN trainer")
    parser.add_argument(
        "--arch",
        type=str,
        default="gin",
        choices=["gin", "gcn", "gat", "both", "all_gnn"],
    )
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.dataset.lower() == "all":
        datasets = DEFAULT_DATASETS
    else:
        datasets = [args.dataset.lower()]

    if args.arch == "both":
        archs = ["gin", "gcn"]
    elif args.arch == "all_gnn":
        archs = ["gin", "gcn", "gat"]
    else:
        archs = [args.arch]

    import gc

    for arch in archs:
        for ds in datasets:
            for seed in seeds:
                try:
                    train_one(
                        ds,
                        seed,
                        arch=arch,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        hidden=args.hidden,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        weight_decay=args.weight_decay,
                        patience=args.patience,
                    )
                except Exception as exc:
                    print(f"[ERROR] arch={arch} dataset={ds} seed={seed}: {exc}", flush=True)
                    import traceback

                    traceback.print_exc()
                finally:
                    # Avoid MPS/RAM accumulation across seeds (prior GAT run died after seed 42)
                    gc.collect()
                    try:
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                    except Exception:
                        pass
                    print(f"[mem] cleaned after arch={arch} ds={ds} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
