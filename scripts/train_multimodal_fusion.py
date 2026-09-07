#!/usr/bin/env python
"""Multimodal late fusion: ChemBERTa (SMILES) + PyG GNN (graph) for Project 1.

Architecture:
  - ChemBERTa encoder → mean-pooled SMILES embedding (frozen then unfrozen)
  - Graph encoder (GIN / GCN / GAT) → graph embedding (global mean pool)
  - Concatenate → MLP regression head (single or multi-task)

Graph-branch width, depth, dropout, and GAT heads match the standalone
PyG models in train_pyg_gnn.py (3 x 128, dropout 0.25, GAT heads=4).
The fusion head and training schedule match the original ChemBERTa+GIN
recipe (20 epochs, freeze 2, text LR 1e-5, graph/head LR 1e-3, batch 16).

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE
    conda activate dc
    cd Project_1
    python scripts/train_multimodal_fusion.py --dataset ppb --seeds 42,43,44 --epochs 20
    python scripts/train_multimodal_fusion.py --dataset ppb --seeds 42,43,44 --graph-encoder gcn
    python scripts/train_multimodal_fusion.py --dataset ppb --seeds 42,43,44 --graph-encoder gat
    python scripts/train_multimodal_fusion.py --dataset ppb --seeds 42 --graph-encoder gcn,gin,gat
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import deepchem as dc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_loaders import DATASET_REGISTRY, DEFAULT_DATA_DIR, DEFAULT_SAVE_DIR  # noqa: E402
from shared.device import get_torch_device  # noqa: E402
from shared.metrics import metrics_to_row, regression_metrics  # noqa: E402
from shared.results_io import append_leaderboard, save_metrics  # noqa: E402
from shared.seed import set_seed  # noqa: E402

DEFAULT_DATASETS = ["ppb", "hopv", "clearance"]
DEFAULT_CHEMBERTA = "seyonec/ChemBERTa-zinc-base-v1"


def load_molgraph(dataset_name: str):
    meta = DATASET_REGISTRY[dataset_name]
    loader_fn = getattr(dc.molnet, meta["loader"])
    save_dir = DEFAULT_SAVE_DIR / dataset_name / "MolGraphConv_scaffold_fusion"
    save_dir.mkdir(parents=True, exist_ok=True)
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=False)
    try:
        tasks, datasets, _ = loader_fn(
            featurizer=featurizer,
            splitter=meta["default_splitter"],
            transformers=["normalization"],
            reload=True,
            data_dir=str(DEFAULT_DATA_DIR),
            save_dir=str(save_dir),
        )
        splitter = meta["default_splitter"]
    except Exception:
        save_dir = DEFAULT_SAVE_DIR / dataset_name / "MolGraphConv_random_fusion"
        tasks, datasets, _ = loader_fn(
            featurizer=featurizer,
            splitter="random",
            transformers=["normalization"],
            reload=True,
            data_dir=str(DEFAULT_DATA_DIR),
            save_dir=str(save_dir),
        )
        splitter = "random"
    return list(tasks), datasets, splitter


def graph_to_pyg(g, y, w) -> Data | None:
    if g is None or not hasattr(g, "node_features") or g.node_features is None:
        return None
    if len(g.node_features) == 0:
        return None
    x = torch.tensor(g.node_features, dtype=torch.float32)
    edge_index = torch.tensor(g.edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    elif edge_index.dim() == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float32).view(-1),
        w=torch.tensor(w, dtype=torch.float32).view(-1),
    )


class FusionMolDataset(Dataset):
    def __init__(self, dc_dataset, tokenizer, max_len: int = 128):
        self.smiles: List[str] = []
        self.graphs: List[Data] = []
        self.ys: List[np.ndarray] = []
        self.ws: List[np.ndarray] = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        X, y, w, ids = dc_dataset.X, np.asarray(dc_dataset.y), np.asarray(dc_dataset.w), dc_dataset.ids
        for i in range(len(X)):
            g = graph_to_pyg(X[i], y[i], w[i])
            if g is None:
                continue
            smi = str(ids[i])
            self.smiles.append(smi)
            self.graphs.append(g)
            self.ys.append(y[i])
            self.ws.append(w[i])

        if not self.smiles:
            raise RuntimeError("No valid multimodal samples")

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
            "graph": self.graphs[idx],
            "labels": torch.tensor(self.ys[idx], dtype=torch.float32).view(-1),
            "weights": torch.tensor(self.ws[idx], dtype=torch.float32).view(-1),
        }


def collate_fusion(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    weights = torch.stack([b["weights"] for b in batch])
    graphs = Batch.from_data_list([b["graph"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "graph": graphs,
        "labels": labels,
        "weights": weights,
    }


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
        self.heads = heads
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


def parse_graph_encoders(spec: str) -> List[str]:
    spec = spec.lower().strip()
    if spec in {"all", "triple", "gcn,gin,gat"}:
        return ["gcn", "gin", "gat"]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    valid = {"gin", "gcn", "gat"}
    unknown = [p for p in parts if p not in valid]
    if unknown:
        raise ValueError(f"Unknown graph encoder(s): {unknown}; use gin, gcn, gat")
    if not parts:
        raise ValueError("Need at least one graph encoder")
    return parts


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        chemberta_name: str,
        n_node_features: int,
        n_tasks: int,
        graph_hidden: int = 128,
        fusion_hidden: int = 256,
        dropout: float = 0.2,
        graph_encoders: List[str] | None = None,
        graph_layers: int = 3,
        graph_dropout: float = 0.25,
        gat_heads: int = 4,
    ):
        super().__init__()
        if graph_encoders is None:
            graph_encoders = ["gin"]
        self.graph_encoders = list(graph_encoders)
        self.text = AutoModel.from_pretrained(chemberta_name)
        text_dim = self.text.config.hidden_size
        self.encoders = nn.ModuleDict(
            {
                arch: GraphEncoder(
                    n_node_features,
                    hidden=graph_hidden,
                    layers=graph_layers,
                    dropout=graph_dropout,
                    arch=arch,
                    heads=gat_heads,
                )
                for arch in self.graph_encoders
            }
        )
        self.head = nn.Sequential(
            nn.Linear(text_dim + graph_hidden * len(self.graph_encoders), fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, n_tasks),
        )

    def freeze_text(self, freeze: bool = True):
        for p in self.text.parameters():
            p.requires_grad = not freeze

    def forward(self, input_ids, attention_mask, graph_batch):
        text_out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        # mean pool over non-pad tokens
        hidden = text_out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        text_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        graph_embs = [
            enc(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            for enc in self.encoders.values()
        ]
        fused = torch.cat([text_emb, *graph_embs], dim=-1)
        return self.head(fused)


@torch.no_grad()
def evaluate(model, loader, device, n_tasks: int) -> dict:
    model.eval()
    preds, labels, weights = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        g = batch["graph"].to(device)
        y = batch["labels"].cpu().numpy()
        w = batch["weights"].cpu().numpy()
        out = model(input_ids, attention_mask, g).cpu().numpy()
        preds.append(out)
        labels.append(y)
        weights.append(w)
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


def train_one(
    dataset_name: str,
    seed: int,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-5,
    graph_lr: float = 1e-3,
    freeze_epochs: int = 2,
    max_len: int = 128,
    chemberta_name: str = DEFAULT_CHEMBERTA,
    graph_encoder: str = "gin",
    graph_hidden: int = 128,
    graph_layers: int = 3,
    graph_dropout: float = 0.25,
    gat_heads: int = 4,
):
    set_seed(seed)
    device = get_torch_device()
    tasks, (train_ds, valid_ds, test_ds), splitter = load_molgraph(dataset_name)
    n_tasks = len(tasks)

    tokenizer = AutoTokenizer.from_pretrained(chemberta_name)
    train_data = FusionMolDataset(train_ds, tokenizer, max_len)
    valid_data = FusionMolDataset(valid_ds, tokenizer, max_len)
    test_data = FusionMolDataset(test_ds, tokenizer, max_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fusion)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fusion)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fusion)

    n_feat = int(train_data.graphs[0].x.shape[1])
    graph_encoders = parse_graph_encoders(graph_encoder)
    encoder_tag = "+".join(a.upper() for a in graph_encoders)
    model = MultimodalFusion(
        chemberta_name,
        n_feat,
        n_tasks,
        graph_hidden=graph_hidden,
        graph_encoders=graph_encoders,
        graph_layers=graph_layers,
        graph_dropout=graph_dropout,
        gat_heads=gat_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(
        f"[{dataset_name}] Multimodal ChemBERTa+{encoder_tag} seed={seed} "
        f"n={len(train_data)}/{len(valid_data)}/{len(test_data)} "
        f"params={n_params} tasks={n_tasks} device={device} "
        f"graph={graph_layers}x{graph_hidden} drop={graph_dropout}"
    )

    if freeze_epochs > 0:
        model.freeze_text(True)
        print(f"  epochs 1-{freeze_epochs}: ChemBERTa frozen")

    def make_optimizer(full: bool):
        if full:
            # differential LR: small for text, larger for graph+head
            params = [
                {"params": model.text.parameters(), "lr": lr},
                {"params": model.encoders.parameters(), "lr": graph_lr},
                {"params": model.head.parameters(), "lr": graph_lr},
            ]
            return torch.optim.AdamW(params, weight_decay=0.01)
        # frozen text
        params = list(model.encoders.parameters()) + list(model.head.parameters())
        return torch.optim.AdamW(params, lr=graph_lr, weight_decay=0.01)

    optimizer = make_optimizer(full=False)
    loss_fn = nn.MSELoss(reduction="none")

    history = []
    best_valid = -1e9
    best_state = None
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.freeze_text(False)
            optimizer = make_optimizer(full=True)
            print(f"  epoch {epoch}: ChemBERTa unfrozen (diff LR)")

        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            g = batch["graph"].to(device)
            y = batch["labels"].to(device)
            w = batch["weights"].to(device)
            out = model(input_ids, attention_mask, g)
            loss = loss_fn(out, y)
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
            f"[{dataset_name}/fusion-{encoder_tag.lower()}] seed={seed} epoch {epoch}/{epochs} "
            f"loss={avg_loss:.4f} train_r2={train_m['r2']:.4f} valid_r2={valid_m['r2']:.4f}"
        )
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        if valid_m["r2"] > best_valid and np.isfinite(valid_m["r2"]):
            best_valid = valid_m["r2"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_m = evaluate(model, train_loader, device, n_tasks)
    valid_m = evaluate(model, valid_loader, device, n_tasks)
    test_m = evaluate(model, test_loader, device, n_tasks)
    train_time = time.time() - t0

    model_tag = f"Multimodal_ChemBERTa_{'_'.join(a.upper() for a in graph_encoders)}_e{epochs}"
    payload = {
        "dataset": dataset_name,
        "model": model_tag,
        "arch": "+".join(graph_encoders),
        "n_params": n_params,
        "seed": seed,
        "splitter": splitter,
        "tasks": tasks,
        "n_tasks": n_tasks,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "lr_text": lr,
        "lr_graph": graph_lr,
        "freeze_epochs": freeze_epochs,
        "batch_size": batch_size,
        "graph_hidden": graph_hidden,
        "graph_layers": graph_layers,
        "graph_dropout": graph_dropout,
        "gat_heads": gat_heads if "gat" in graph_encoders else None,
        "chemberta": chemberta_name,
        "device": str(device),
        "train_time_sec": round(train_time, 3),
        "best_valid_r2": best_valid,
        "train": train_m,
        "valid": valid_m,
        "test": test_m,
        "history": history,
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
                "freeze_epochs": freeze_epochs,
                "hidden": graph_hidden,
                "arch": "+".join(graph_encoders),
            },
        )
    )
    print(
        f"[{dataset_name}] Multimodal ChemBERTa+{encoder_tag} seed={seed} "
        f"best_epoch={best_epoch} test R2={test_m['r2']:.4f} RMSE={test_m['rmse']:.4f} "
        f"({train_time:.1f}s)"
    )
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5, help="ChemBERTa LR when unfrozen")
    p.add_argument("--graph-lr", type=float, default=1e-3)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--model-name", default=DEFAULT_CHEMBERTA)
    p.add_argument(
        "--graph-encoder",
        default="gin",
        help="Graph branch: gin, gcn, gat, or comma list (e.g. gcn,gin,gat)",
    )
    p.add_argument("--graph-hidden", type=int, default=128)
    p.add_argument("--graph-layers", type=int, default=3)
    p.add_argument("--graph-dropout", type=float, default=0.25)
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
                    graph_lr=args.graph_lr,
                    freeze_epochs=args.freeze_epochs,
                    max_len=args.max_len,
                    chemberta_name=args.model_name,
                    graph_encoder=args.graph_encoder,
                    graph_hidden=args.graph_hidden,
                    graph_layers=args.graph_layers,
                    graph_dropout=args.graph_dropout,
                    gat_heads=args.gat_heads,
                )
            except Exception as e:
                print(f"[ERROR] {ds} {args.graph_encoder} seed={seed}: {e}")
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
