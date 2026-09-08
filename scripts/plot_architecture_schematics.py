#!/usr/bin/env python
"""Publication-quality architecture schematics for the PPB fusion models.

Writes vector PDF and 300 dpi PNG.

Usage:
    conda activate dc
    python scripts/plot_architecture_schematics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "figures",
]

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
INDIGO = "#332288"
GREY = "#F4F4F4"
EDGE = "#222222"
TEXT = "#111111"


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save(fig, stem: str):
    for outdir in OUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
        pdf = outdir / f"{stem}.pdf"
        png = outdir / f"{stem}.png"
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        print(f"Wrote {pdf}")
        print(f"Wrote {png}")
    plt.close(fig)


def box(ax, cx, cy, w, h, text, fc=GREY, fs=8.2, weight="normal"):
    x, y = cx - w / 2.0, cy - h / 2.0
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.035",
        facecolor=fc,
        edgecolor=EDGE,
        linewidth=0.95,
        mutation_aspect=0.8,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=TEXT,
        linespacing=1.28,
        fontweight=weight,
        wrap=False,
    )
    return {"cx": cx, "cy": cy, "w": w, "h": h, "top": cy + h / 2, "bot": cy - h / 2, "left": cx - w / 2, "right": cx + w / 2}


def arrow(ax, x1, y1, x2, y2, lw=1.15):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=EDGE,
            lw=lw,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
            capstyle="butt",
        ),
    )


def v_arrow(ax, a, b):
    arrow(ax, a["cx"], a["bot"] - 0.003, b["cx"], b["top"] + 0.003)


def h_arrow(ax, a, b):
    arrow(ax, a["right"] + 0.003, a["cy"], b["left"] - 0.003, b["cy"])


def stack_centers(n, h, gap, y_top):
    """Centers of n equal-height boxes with equal edge-to-edge gap, from y_top down."""
    return [y_top - h / 2.0 - i * (h + gap) for i in range(n)]


def footer(ax, last_box, text, gap=0.07):
    y = last_box["bot"] - gap
    ax.text(0.50, y, text, ha="center", va="top", fontsize=7.4, color="#444444")
    return y


def finish_axes(ax, last_box, footer_y, top=1.02):
    ax.set_xlim(0, 1)
    ax.set_ylim(footer_y - 0.045, top)
    ax.axis("off")


def plot_fusion():
    h, gap = 0.078, 0.050
    y_top = 0.90
    ys = stack_centers(8, h, gap, y_top)

    fig, ax = plt.subplots(figsize=(10.4, 9.4))
    ax.set_title("Late fusion: ChemBERTa + one GNN", fontsize=12, fontweight="bold", pad=10, color=TEXT)
    ax.text(0.25, y_top + 0.055, "SMILES branch", ha="center", va="center", fontsize=9.5, fontweight="bold", color=BLUE)
    ax.text(0.75, y_top + 0.055, "Graph branch", ha="center", va="center", fontsize=9.5, fontweight="bold", color=ORANGE)

    bw = 0.36
    s = [
        box(ax, 0.25, ys[0], bw, h, "SMILES string", fc="#D6EAF8"),
        box(ax, 0.25, ys[1], bw, h, "Tokenizer\nmax length 128", fc="#D6EAF8"),
        box(ax, 0.25, ys[2], bw, h, "ChemBERTa encoder\n768-d token states", fc="#AED6F1"),
        box(ax, 0.25, ys[3], bw, h, "Mean pool non-pad tokens\n768-d molecule vector", fc="#D6EAF8"),
    ]
    g = [
        box(ax, 0.75, ys[0], bw, h, "Molecular graph", fc="#FDEBD0"),
        box(ax, 0.75, ys[1], bw, h, "Atom features\n~30-d per atom", fc="#FDEBD0"),
        box(ax, 0.75, ys[2], bw, h, "GNN encoder  3 x 128\nGIN or GCN or GAT", fc="#FAD7A0"),
        box(ax, 0.75, ys[3], bw, h, "Global mean pool\n128-d molecule vector", fc="#FDEBD0"),
    ]
    for a, b in zip(s, s[1:]):
        v_arrow(ax, a, b)
    for a, b in zip(g, g[1:]):
        v_arrow(ax, a, b)

    cw = 0.44
    cat = box(ax, 0.50, ys[4], cw, h, "Concatenate   896-d", fc="#D5F5E3", fs=9, weight="bold")
    m1 = box(ax, 0.50, ys[5], cw, h, "Dense 256  + ReLU  + dropout 0.2", fc="#E8DAEF")
    m2 = box(ax, 0.50, ys[6], cw, h, "Dense 128  + ReLU  + dropout 0.2", fc="#E8DAEF")
    out = box(ax, 0.50, ys[7], cw, h, "PPB prediction   1-d", fc="#D7BDE2", fs=9, weight="bold")
    arrow(ax, s[3]["cx"], s[3]["bot"] - 0.003, cat["left"] + 0.08, cat["top"] + 0.003)
    arrow(ax, g[3]["cx"], g[3]["bot"] - 0.003, cat["right"] - 0.08, cat["top"] + 0.003)
    v_arrow(ax, cat, m1)
    v_arrow(ax, m1, m2)
    v_arrow(ax, m2, out)

    fy = footer(
        ax,
        out,
        "Supplementary Figure 2. ChemBERTa+GNN. Same head for ChemBERTa+GIN, ChemBERTa+GCN, and ChemBERTa+GAT.  ~44.5M parameters (mostly ChemBERTa).",
    )
    finish_axes(ax, out, fy, top=y_top + 0.10)
    _save(fig, "figS2_arch_chemberta_gnn")


def plot_triple():
    h, gap = 0.078, 0.050
    y_top = 0.92
    ys = stack_centers(8, h, gap, y_top)

    fig, ax = plt.subplots(figsize=(10.4, 9.4))
    ax.set_title("Graph-only concat control: GCN + GIN + GAT", fontsize=12, fontweight="bold", pad=10, color=TEXT)

    top = box(ax, 0.50, ys[0], 0.44, h, "Molecular graph", fc="#FDEBD0", fs=9, weight="bold")
    feat = box(ax, 0.50, ys[1], 0.44, h, "Atom features   ~30-d per atom", fc="#FDEBD0")
    v_arrow(ax, top, feat)

    cols = [
        (0.20, "GCN\n3 x 128", "#D6EAF8", "#AED6F1"),
        (0.50, "GIN\n3 x 128", "#D5F5E3", "#ABEBC6"),
        (0.80, "GAT\n3 x 128, 4 heads", "#F5EEF8", "#D7BDE2"),
    ]
    pools = []
    for x, label, fc1, fc2 in cols:
        e = box(ax, x, ys[2], 0.26, h, label, fc=fc2, fs=8.4, weight="bold")
        p = box(ax, x, ys[3], 0.26, h, "Mean pool\n128-d", fc=fc1, fs=8.0)
        arrow(ax, feat["cx"], feat["bot"] - 0.003, e["cx"], e["top"] + 0.003)
        v_arrow(ax, e, p)
        pools.append(p)

    cat = box(ax, 0.50, ys[4], 0.46, h, "Concatenate   384-d", fc="#FAD7A0", fs=9, weight="bold")
    for p in pools:
        arrow(ax, p["cx"], p["bot"] - 0.003, cat["cx"] + (p["cx"] - 0.50) * 0.28, cat["top"] + 0.003)
    m1 = box(ax, 0.50, ys[5], 0.46, h, "Dense 256  + ReLU  + dropout 0.2", fc="#E8DAEF")
    m2 = box(ax, 0.50, ys[6], 0.46, h, "Dense 128  + ReLU  + dropout 0.2", fc="#E8DAEF")
    out = box(ax, 0.50, ys[7], 0.46, h, "PPB prediction   1-d", fc="#D7BDE2", fs=9, weight="bold")
    v_arrow(ax, cat, m1)
    v_arrow(ax, m1, m2)
    v_arrow(ax, m2, out)

    fy = footer(
        ax,
        out,
        "Supplementary Figure 3. GCN+GIN+GAT. No ChemBERTa. Graph stacks match the standalone GCN, GIN, and GAT.  ~0.35M parameters.",
    )
    finish_axes(ax, out, fy, top=y_top + 0.08)
    _save(fig, "figS3_arch_triple_gnn")


def plot_schedule():
    w, h, gap = 0.20, 0.46, 0.045
    n = 4
    total = n * w + (n - 1) * gap
    x0 = 0.50 - total / 2.0 + w / 2.0
    xs = [x0 + i * (w + gap) for i in range(n)]
    cy = 0.56

    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    ax.set_title("ChemBERTa + GNN training schedule", fontsize=12, fontweight="bold", pad=8, color=TEXT)

    specs = [
        ("Epochs 1-2\n\nChemBERTa frozen\nTrain GNN + MLP\nlr 1e-3", "#D6EAF8"),
        ("Epochs 3-20\n\nUnfreeze ChemBERTa\nText lr 1e-5\nGraph / head lr 1e-3", "#FDEBD0"),
        ("Selection\n\nKeep checkpoint\nwith highest\nvalidation R2", "#D5F5E3"),
        ("Test\n\nScore the\ntest set\nonce", "#E8DAEF"),
    ]
    boxes = [box(ax, x, cy, w, h, text, fc=fc, fs=8.0) for x, (text, fc) in zip(xs, specs)]
    for a, b in zip(boxes, boxes[1:]):
        h_arrow(ax, a, b)

    fy = footer(
        ax,
        boxes[0],
        "Supplementary Figure 4. ChemBERTa+GNN training schedule. Batch size 16. AdamW, weight decay 0.01, gradient clip 1.0. Same schedule for GIN, GCN, and GAT partners.",
        gap=0.08,
    )
    finish_axes(ax, boxes[0], fy, top=0.98)
    _save(fig, "figS4_arch_fusion_schedule")


def plot_standalone_gnn():
    h, gap = 0.078, 0.050
    y_top = 0.92
    labels = [
        ("Molecular graph", GREY),
        ("Atom features  ~30-d\nno bond attributes", "#FDEBD0"),
        ("Message-passing layer 1\nhidden 128, BN, ReLU, dropout 0.25", "#D6EAF8"),
        ("Message-passing layer 2\nhidden 128, BN, ReLU, dropout 0.25", "#D6EAF8"),
        ("Message-passing layer 3\nhidden 128, BN, ReLU, dropout 0.25", "#AED6F1"),
        ("Global mean pool  128-d", "#D5F5E3"),
        ("Linear readout  to  PPB  1-d", "#D7BDE2"),
    ]
    ys = stack_centers(len(labels), h, gap, y_top)

    fig, ax = plt.subplots(figsize=(8.6, 8.8))
    ax.set_title("Standalone graph encoder (GCN, GIN, or GAT)", fontsize=12, fontweight="bold", pad=10, color=TEXT)
    boxes = [
        box(ax, 0.50, y, 0.62, h, text, fc=fc, fs=8.4) for y, (text, fc) in zip(ys, labels)
    ]
    for a, b in zip(boxes, boxes[1:]):
        v_arrow(ax, a, b)
    fy = footer(
        ax,
        boxes[-1],
        "Supplementary Figure 1. Standalone GNN (GCN, GIN, or GAT). GCN: degree-normalized mean.  GIN: sum then MLP.  GAT: 4 heads (concat, then mean on last layer).",
    )
    finish_axes(ax, boxes[-1], fy, top=y_top + 0.08)
    _save(fig, "figS1_arch_standalone_gnn")


def main():
    _style()
    plot_standalone_gnn()
    plot_fusion()
    plot_triple()
    plot_schedule()
    print("Done.")


if __name__ == "__main__":
    main()
