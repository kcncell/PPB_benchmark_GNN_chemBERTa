# Supplementary files

Architecture schematics cited in the manuscript as Supplementary Figures S1–S4.
Rebuild with:

```bash
conda activate dc
python scripts/plot_architecture_schematics.py
```

Files live under `figures/` (and copies under `PPB_pivot/figures/architecture/`).

| Manuscript | File stem |
|---|---|
| Supplementary Figure S1 | `figures/figS0_arch_standalone_gnn` |
| Supplementary Figure S2 | `figures/figS1_arch_chemberta_gnn` |
| Supplementary Figure S3 | `figures/figS3_arch_fusion_schedule` |
| Supplementary Figure S4 | `figures/figS2_arch_triple_gnn` |

Each stem has `.pdf` (vector), `.png` (300 dpi), and `_600dpi.png`.

Mermaid source drafts used while designing the schematics:

- `PPB_pivot/figures/excalidraw_fusion_chemberta_gnn.mmd`
- `PPB_pivot/figures/excalidraw_fusion_training.mmd`
- `PPB_pivot/figures/excalidraw_fusion_triple_gnn.mmd`
