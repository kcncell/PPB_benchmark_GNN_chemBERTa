#!/usr/bin/env python
"""Publication figures for the PPB-only manuscript.

Soft Okabe-Ito palette (colorblind-aware), matching Prostate_Cancer_FieldEffect
manuscript theme (00_theme.R): classic axes, no grid, uniform fonts, legend outside.

Usage:
    conda activate dc
    cd Project_1
    python scripts/plot_publication_figures.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
DEFAULT_OUT = PROJECT_ROOT / "figures"

# Soft Okabe-Ito inspired (colorblind-aware; from Prostate 00_theme.R).
# Yellow (#F0E442) avoided for bars on white; use grey as last accent.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange/gold
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#332288",  # indigo
    "#999999",  # grey
    "#882255",  # wine
    "#44AA99",  # teal
]

MODEL_ORDER = [
    "RandomForest_ECFP",
    "XGBoost_ECFP",
    "ChemBERTa_zinc_base_v1_lr1e-05_e20",
    "PyG_GIN_h128_e100",
    "PyG_GCN_h128_e100",
    "PyG_GAT_h128_e100",
    "Multimodal_ChemBERTa_GIN_e20",
    "Multimodal_ChemBERTa_GCN_e20",
    "Multimodal_ChemBERTa_GAT_e20",
    "TripleGNN_GCN_GIN_GAT_h128_e100",
]
MODEL_LABELS = {
    "RandomForest_ECFP": "RF + ECFP",
    "XGBoost_ECFP": "XGBoost + ECFP",
    "ChemBERTa_zinc_base_v1_lr1e-05_e20": "ChemBERTa",
    "PyG_GIN_h128_e100": "GIN",
    "PyG_GCN_h128_e100": "GCN",
    "PyG_GAT_h128_e100": "GAT",
    "Multimodal_ChemBERTa_GIN_e20": "ChemBERTa+GIN",
    "Multimodal_ChemBERTa_GCN_e20": "ChemBERTa+GCN",
    "Multimodal_ChemBERTa_GAT_e20": "ChemBERTa+GAT",
    "TripleGNN_GCN_GIN_GAT_h128_e100": "GCN+GIN+GAT",
    "RF_conformal": "RF + conformal",
    "RF_ensemble5_conformal": "RF ensemble + conformal",
}
DATASET_ORDER = ["ppb"]
# Short axis labels (full names in captions); consistent capitalization
DATASET_LABELS = {
    "ppb": "PPB",
}
DATASET_LABELS_FULL = {
    "ppb": "Plasma Protein Binding (PPB)",
}

COLORS = {m: OKABE_ITO[i % len(OKABE_ITO)] for i, m in enumerate(MODEL_ORDER)}
SEED_COLORS = {42: "#0072B2", 43: "#E69F00", 44: "#D55E00"}
SEED_MARKERS = {42: "o", 43: "s", 44: "^"}

# Font sizes aligned with Prostate 00_theme.R (classic, no grid)
SIZE_TITLE = 12
SIZE_SUBTITLE = 10
SIZE_AXIS = 11
SIZE_TICK = 9
SIZE_LEGEND = 8
SIZE_ANNOT = 8


def _setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": SIZE_TICK,
            "axes.titlesize": SIZE_TITLE,
            "axes.labelsize": SIZE_AXIS,
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "legend.fontsize": SIZE_LEGEND,
            "xtick.labelsize": SIZE_TICK,
            "ytick.labelsize": SIZE_TICK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.0,
            "mathtext.default": "regular",
        }
    )


def _legend_below(ax_or_fig, handles=None, labels=None, ncol=4, y=-0.14, ax=None):
    """Place a clean multi-column legend below the axes (no overlap)."""
    kwargs = dict(
        frameon=False,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.4,
        fontsize=SIZE_LEGEND,
        borderaxespad=0.0,
    )
    if handles is not None and labels is not None:
        if ax is not None:
            ax.legend(handles, labels, **kwargs)
        else:
            ax_or_fig.legend(handles, labels, **kwargs)
    else:
        ax_or_fig.legend(**kwargs)


def _save(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.08)
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {outpath}")


def load_leaderboard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["dataset"].isin(DATASET_ORDER)].copy()
    df = df[~df["model"].astype(str).str.contains(r"_e5$|_e3$|conformal|GCN_GIN_GAT_e20")].copy()
    return df


def summarize(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    sub = df[df["model"].isin(models)].copy()
    return (
        sub.groupby(["dataset", "model"], as_index=False)
        .agg(
            n=("r2", "count"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            train_r2_mean=("train_r2", "mean"),
            valid_r2_mean=("valid_r2", "mean"),
        )
    )


def plot_grouped_bars(
    summary: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    outpath: Path,
    models: List[str],
):
    datasets = [d for d in DATASET_ORDER if d in summary["dataset"].unique()]
    models = [m for m in models if m in summary["model"].unique()]
    n_ds, n_m = len(datasets), len(models)
    x = np.arange(n_ds, dtype=float)
    width = min(0.12, 0.8 / max(n_m, 1))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, m in enumerate(models):
        means, stds = [], []
        for d in datasets:
            row = summary[(summary.dataset == d) & (summary.model == m)]
            if len(row):
                means.append(float(row[metric_mean].iloc[0]))
                s = row[metric_std].iloc[0]
                stds.append(0.0 if pd.isna(s) else float(s))
            else:
                means.append(np.nan)
                stds.append(0.0)
        offset = (i - (n_m - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=2.5,
            label=MODEL_LABELS.get(m, m),
            color=COLORS.get(m, OKABE_ITO[i % len(OKABE_ITO)]),
            edgecolor="#333333",
            linewidth=0.35,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
            zorder=3,
        )

    ax.axhline(0, color="#666666", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("Dataset")
    _legend_below(ax, ncol=min(4, n_m), y=-0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    _save(fig, outpath)


def plot_heatmap(summary: pd.DataFrame, outpath: Path, models: List[str]):
    datasets = [d for d in DATASET_ORDER if d in summary["dataset"].unique()]
    models = [m for m in models if m in summary["model"].unique()]
    mat = np.full((len(datasets), len(models)), np.nan)
    for i, d in enumerate(datasets):
        for j, m in enumerate(models):
            row = summary[(summary.dataset == d) & (summary.model == m)]
            if len(row):
                mat[i, j] = float(row["r2_mean"].iloc[0])

    # Purple-white-green diverging (from 00_theme.R heat scale idea)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "soft_div",
        ["#5E3C99", "#F7F7F7", "#00441B"],
    )
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-0.15, vmax=0.25)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                # dark text on light cells
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#111111",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"Mean test $R^{2}$", fontsize=SIZE_LEGEND)
    cbar.ax.tick_params(labelsize=SIZE_TICK - 1)
    ax.set_title(r"Mean test $R^{2}$ by model and dataset", fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, outpath)


def plot_generalization_gap(df: pd.DataFrame, outpath: Path, models: List[str]):
    sub = df[df["model"].isin(models)].copy()
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for m in models:
        part = sub[sub.model == m]
        if part.empty:
            continue
        ax.scatter(
            part["train_r2"],
            part["r2"],
            c=COLORS.get(m, "#999999"),
            s=42,
            alpha=0.9,
            edgecolors="#333333",
            linewidths=0.3,
            label=MODEL_LABELS.get(m, m),
            zorder=3,
        )
    ax.plot([-0.2, 1.05], [-0.2, 1.05], color="#666666", ls="--", lw=0.9, zorder=1, label="Train = test")
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.25, 0.55)
    ax.set_xlabel(r"Train $R^{2}$")
    ax.set_ylabel(r"Test $R^{2}$")
    ax.set_title("PPB generalization gap (one point per seed)", fontweight="bold", pad=8)
    _legend_below(ax, ncol=4, y=-0.18)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    _save(fig, outpath)


def load_histories(results_root: Path) -> pd.DataFrame:
    rows = []
    for path in results_root.rglob("metrics.json"):
        # ADME only
        parts = path.parts
        if "hopv" in parts or "hppb" in parts:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        history = data.get("history")
        if not history:
            continue
        dataset = data.get("dataset") or path.parts[-4]
        if dataset not in DATASET_ORDER:
            continue
        model = data.get("model") or path.parts[-3]
        seed = data.get("seed")
        if seed is None:
            m = re.search(r"seed(\d+)", path.as_posix())
            seed = int(m.group(1)) if m else -1
        for h in history:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": int(seed),
                    "epoch": h.get("epoch"),
                    "loss": h.get("loss"),
                    "train_r2": h.get("train_r2"),
                    "valid_r2": h.get("valid_r2"),
                }
            )
    return pd.DataFrame(rows)


def _model_family(model: str) -> Optional[str]:
    s = str(model)
    if "Multimodal" in s:
        return None
    if "ChemBERTa" in s:
        return "ChemBERTa"
    if "GIN" in s:
        return "GIN"
    if "GCN" in s:
        return "GCN"
    if "GAT" in s:
        return "GAT"
    return None


def plot_learning_curves_facet(hist: pd.DataFrame, outpath: Path):
    if hist.empty:
        print("No histories; skip learning curves.")
        return
    hist = hist.copy()
    hist["family"] = hist["model"].map(_model_family)
    hist = hist.dropna(subset=["family"])
    families = [f for f in ["ChemBERTa", "GIN", "GCN", "GAT"] if f in set(hist["family"])]
    datasets = [d for d in DATASET_ORDER if d in hist["dataset"].unique()]
    if not datasets or not families:
        print("No learning-curve families/datasets; skip.")
        return

    fig, axes = plt.subplots(
        len(families),
        len(datasets),
        figsize=(6.8 if len(datasets) == 1 else 3.5 * len(datasets), 2.05 * len(families)),
        sharex=False,
        sharey=False,
    )
    if len(families) == 1:
        axes = np.array([axes])
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)

    for i, fam in enumerate(families):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            sub = hist[(hist.family == fam) & (hist.dataset == ds)]
            if sub.empty:
                ax.set_visible(False)
                continue
            for seed in sorted(sub.seed.unique()):
                s = sub[sub.seed == seed].sort_values("epoch")
                if s["loss"].notna().any():
                    ax.plot(
                        s.epoch,
                        s.loss,
                        color=SEED_COLORS.get(int(seed), "#999999"),
                        marker=SEED_MARKERS.get(int(seed), "o"),
                        markevery=max(1, len(s) // 6),
                        markersize=3.5,
                        lw=1.3,
                        label=f"Seed {seed}",
                    )
            if i == 0:
                ax.set_title(DATASET_LABELS_FULL[ds], fontweight="bold", fontsize=SIZE_TITLE - 1)
            if j == 0:
                ax.set_ylabel(f"{fam}\nTraining loss", fontsize=SIZE_AXIS - 1)
            if i == len(families) - 1:
                ax.set_xlabel("Epoch", fontsize=SIZE_AXIS - 1)
            ax.tick_params(labelsize=SIZE_TICK - 1)

    # Single shared legend below all panels
    handles, labels = [], []
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=SIZE_LEGEND,
        )
    fig.suptitle("PPB training loss by model family", fontweight="bold", y=1.0, fontsize=SIZE_TITLE)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    _save(fig, outpath)


def plot_uq_bars(df: pd.DataFrame, outpath: Path):
    uq = df[
        df["model"].astype(str).str.contains("conformal|ensemble", case=False, regex=True)
        & df["dataset"].isin(DATASET_ORDER)
    ].copy()
    if uq.empty or "picp" not in uq.columns:
        print("No UQ rows; skip UQ figure.")
        return
    uq = uq.dropna(subset=["picp"])
    if uq.empty:
        return

    g = uq.groupby(["dataset", "model"], as_index=False).agg(
        picp_mean=("picp", "mean"),
        picp_std=("picp", "std"),
        mpiw_mean=("mpiw", "mean") if "mpiw" in uq.columns else ("picp", "mean"),
        mpiw_std=("mpiw", "std") if "mpiw" in uq.columns else ("picp", "std"),
    )
    datasets = [d for d in DATASET_ORDER if d in g.dataset.unique()]
    models = sorted(g.model.unique())
    x = np.arange(len(datasets), dtype=float)
    width = 0.35 if len(models) <= 2 else 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ax, metric, ylabel, title, target in [
        (axes[0], "picp_mean", "PICP (empirical coverage)", "Coverage (PICP)", 0.9),
        (axes[1], "mpiw_mean", "MPIW (normalized scale)", "Mean interval width", None),
    ]:
        for i, m in enumerate(models):
            vals, errs = [], []
            for d in datasets:
                row = g[(g.dataset == d) & (g.model == m)]
                if len(row):
                    vals.append(float(row[metric].iloc[0]))
                    std_col = "picp_std" if metric == "picp_mean" else "mpiw_std"
                    s = row[std_col].iloc[0]
                    errs.append(0.0 if pd.isna(s) else float(s))
                else:
                    vals.append(np.nan)
                    errs.append(0.0)
            offset = (i - (len(models) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                vals,
                width=width * 0.9,
                yerr=errs,
                capsize=2.5,
                color=OKABE_ITO[i % len(OKABE_ITO)],
                edgecolor="#333333",
                linewidth=0.35,
                label=MODEL_LABELS.get(m, m.replace("_", " ")),
                error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
            )
        if target is not None:
            ax.axhline(target, color="#D55E00", ls="--", lw=1.0, label="Target 0.90")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
        ax.set_xlabel("Dataset")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=6)
        ax.tick_params(labelsize=SIZE_TICK)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    # Also pull MPIW-panel-only handles if needed
    h1, l1 = axes[1].get_legend_handles_labels()
    for h, l in zip(h1, l1):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    fig.legend(
        h2,
        l2,
        frameon=False,
        loc="lower center",
        ncol=min(4, len(l2)),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=SIZE_LEGEND,
    )
    fig.suptitle("Uncertainty quantification (split conformal)", fontweight="bold", y=1.01, fontsize=SIZE_TITLE)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, outpath)


def plot_delta_vs_rf(summary: pd.DataFrame, outpath: Path, models: List[str]):
    datasets = [d for d in DATASET_ORDER if d in summary.dataset.unique()]
    rf_name = "RandomForest_ECFP"
    deltas, names = [], []
    for d in datasets:
        part = summary[(summary.dataset == d) & (summary.model.isin(models))]
        if part.empty:
            deltas.append(0.0)
            names.append("?")
            continue
        best = part.loc[part.r2_mean.idxmax()]
        rf = part[part.model == rf_name]
        rf_r2 = float(rf.r2_mean.iloc[0]) if len(rf) else 0.0
        deltas.append(float(best.r2_mean) - rf_r2)
        names.append(MODEL_LABELS.get(best.model, best.model))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    x = np.arange(len(datasets))
    cols = ["#009E73" if v >= 0 else "#D55E00" for v in deltas]
    bars = ax.bar(x, deltas, color=cols, edgecolor="#333333", linewidth=0.4, width=0.55)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_xlabel("Dataset")
    ax.set_ylabel(r"$\Delta$ test $R^{2}$ (best model $-$ RF)")
    ax.set_title("Improvement over random forest baseline", fontweight="bold", pad=8)
    for b, name, v in zip(bars, names, deltas):
        ypos = b.get_height()
        va = "bottom" if v >= 0 else "top"
        offset = 0.01 if v >= 0 else -0.01
        ax.text(
            b.get_x() + b.get_width() / 2,
            ypos + offset,
            f"{name}\n{v:+.3f}",
            ha="center",
            va=va,
            fontsize=SIZE_ANNOT,
        )
    # Soft room for labels above bars
    ymax = max(deltas) if deltas else 0.1
    ymin = min(0.0, min(deltas) if deltas else 0.0)
    pad = 0.06 * max(abs(ymax), abs(ymin), 0.1)
    ax.set_ylim(ymin - pad, ymax + pad * 2.2)
    fig.tight_layout()
    _save(fig, outpath)


def plot_ppb_model_bars(
    summary: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    outpath: Path,
    models: List[str],
    hline: Optional[float] = 0.0,
):
    models = [m for m in models if m in summary["model"].unique()]
    means, stds = [], []
    for m in models:
        row = summary[summary.model == m]
        means.append(float(row[metric_mean].iloc[0]))
        s = row[metric_std].iloc[0]
        stds.append(0.0 if pd.isna(s) else float(s))
    x = np.arange(len(models), dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=2.5,
        color=[COLORS.get(m, OKABE_ITO[i % len(OKABE_ITO)]) for i, m in enumerate(models)],
        edgecolor="#333333",
        linewidth=0.35,
        error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
        zorder=3,
        width=0.72,
    )
    if hline is not None:
        ax.axhline(hline, color="#666666", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=32, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, outpath)


def plot_ppb_seed_dots(df: pd.DataFrame, outpath: Path, models: List[str]):
    models = [m for m in models if m in df["model"].unique()]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(models), dtype=float)
    for i, m in enumerate(models):
        part = df[df.model == m].sort_values("seed")
        for _, row in part.iterrows():
            ax.scatter(
                i,
                row["r2"],
                c=SEED_COLORS.get(int(row["seed"]), "#999999"),
                marker=SEED_MARKERS.get(int(row["seed"]), "o"),
                s=48,
                edgecolors="#333333",
                linewidths=0.35,
                zorder=3,
            )
        if len(part):
            ax.hlines(
                part["r2"].mean(),
                i - 0.28,
                i + 0.28,
                colors="#111111",
                linewidth=1.1,
                zorder=2,
            )
    for seed, color in SEED_COLORS.items():
        ax.scatter([], [], c=color, marker=SEED_MARKERS[seed], s=48, edgecolors="#333333", linewidths=0.35, label=f"Seed {seed}")
    ax.axhline(0, color="#666666", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=32, ha="right")
    ax.set_ylabel(r"Test $R^{2}$")
    ax.set_title("Per-seed test $R^{2}$ on PPB (scaffold split)", fontweight="bold", pad=8)
    _legend_below(ax, ncol=3, y=-0.22)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    _save(fig, outpath)


def plot_delta_all_vs_rf(summary: pd.DataFrame, outpath: Path, models: List[str]):
    rf_name = "RandomForest_ECFP"
    rf = summary[summary.model == rf_name]
    if rf.empty:
        print("No RF row; skip delta plot.")
        return
    rf_r2 = float(rf.r2_mean.iloc[0])
    models = [m for m in models if m in summary.model.unique() and m != rf_name]
    deltas = []
    for m in models:
        row = summary[summary.model == m]
        deltas.append(float(row.r2_mean.iloc[0]) - rf_r2)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(models), dtype=float)
    cols = ["#009E73" if v >= 0 else "#D55E00" for v in deltas]
    ax.bar(x, deltas, color=cols, edgecolor="#333333", linewidth=0.35, width=0.72)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=32, ha="right")
    ax.set_ylabel(r"$\Delta$ test $R^{2}$ versus RF + ECFP")
    ax.set_title("Mean test $R^{2}$ relative to random forest", fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, outpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--leaderboard", type=str, default=str(RESULTS / "leaderboard.csv"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_style()

    df = load_leaderboard(Path(args.leaderboard))
    models = [m for m in MODEL_ORDER if m in df.model.unique()]
    summary = summarize(df, models)
    # Keep only complete three-seed rows in the paper figures.
    if "n" in summary.columns:
        summary = summary[summary["n"] >= 3].copy()
        models = [m for m in models if m in set(summary.model)]
    summary.to_csv(outdir / "figure_source_summary_ppb.csv", index=False)

    plot_ppb_model_bars(
        summary,
        "r2_mean",
        "r2_std",
        ylabel=r"Test $R^{2}$ (mean $\pm$ std)",
        title=r"PPB test $R^{2}$ under one Bemis--Murcko scaffold split",
        outpath=outdir / "fig1_test_r2_bars.png",
        models=models,
    )
    plot_ppb_model_bars(
        summary,
        "rmse_mean",
        "rmse_std",
        ylabel="Test RMSE (normalized labels)",
        title="PPB test RMSE under one Bemis--Murcko scaffold split",
        outpath=outdir / "fig2_test_rmse_bars.png",
        models=models,
        hline=None,
    )
    plot_ppb_seed_dots(df[df.model.isin(models)], outdir / "fig3_seed_r2.png", models)
    plot_generalization_gap(df, outdir / "fig4_generalization_gap.png", models)

    hist = load_histories(RESULTS)
    if not hist.empty:
        hist = hist[hist["dataset"] == "ppb"].copy()
        hist.to_csv(outdir / "figure_source_histories_ppb.csv", index=False)
        plot_learning_curves_facet(hist, outdir / "fig5_loss_curves_all.png")

    plot_delta_all_vs_rf(summary, outdir / "fig7_delta_vs_rf.png", models)

    print(f"\nAll figures in: {outdir}")


if __name__ == "__main__":
    main()
