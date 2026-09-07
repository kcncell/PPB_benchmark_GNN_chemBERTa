#!/usr/bin/env bash
# ChemBERTa + GCN + GIN + GAT late fusion. One seed at a time (MPS memory).
# Same schedule as ChemBERTa+GNN: 20 epochs, freeze 2, text LR 1e-5,
# graph/head LR 1e-3, batch 16.
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/results"
PIVOT="$ROOT/PPB_pivot"
mkdir -p "$LOGDIR" "$PIVOT"
MASTER="$LOGDIR/multimodal_quad_ppb.log"
SEEDS="${SEEDS:-42}"

# shellcheck disable=SC1091
source "$ROOT/scripts/activate_dc.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

log "=== START ChemBERTa+GCN+GIN+GAT PPB seeds ${SEEDS} ==="

python scripts/train_multimodal_fusion.py \
  --dataset ppb \
  --seeds "${SEEDS}" \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-5 \
  --graph-lr 1e-3 \
  --freeze-epochs 2 \
  --graph-encoder gcn,gin,gat \
  --graph-hidden 128 \
  --graph-layers 3 \
  --graph-dropout 0.25 \
  --gat-heads 4 \
  2>&1 | tee -a "$MASTER"

log "=== quad fusion done; compiling PPB_pivot ==="
python scripts/compile_ppb_pivot.py 2>&1 | tee -a "$MASTER"
log "=== DONE ChemBERTa+GCN+GIN+GAT PPB seeds ${SEEDS} ==="
