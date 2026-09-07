#!/usr/bin/env bash
# GCN + GIN + GAT late fusion on PPB, no ChemBERTa. Control for ChemBERTa+GNN.
# Graph stacks match standalone GNNs (3 x 128, drop 0.25, GAT heads=4).
# Schedule matches standalone GNNs: 100 epochs, patience 25, Adam 1e-3, wd 1e-4.
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/results"
PIVOT="$ROOT/PPB_pivot"
mkdir -p "$LOGDIR" "$PIVOT"
MASTER="$LOGDIR/triple_gnn_ppb.log"

# shellcheck disable=SC1091
source "$ROOT/scripts/activate_dc.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

log "=== START TripleGNN GCN+GIN+GAT PPB seeds ${SEEDS:-42,43,44} ==="

PYTHONUNBUFFERED=1 python scripts/train_gnn_triple_fusion.py \
  --dataset ppb \
  --seeds "${SEEDS:-42,43,44}" \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --hidden 128 \
  --num-layers 3 \
  --dropout 0.25 \
  --weight-decay 1e-4 \
  --patience 25 \
  --gat-heads 4 \
  2>&1 | tee -a "$MASTER"

log "=== TripleGNN done; compiling PPB_pivot ==="
python scripts/compile_ppb_pivot.py 2>&1 | tee -a "$MASTER"
log "=== DONE TripleGNN PPB ==="
