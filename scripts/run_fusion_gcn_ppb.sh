#!/usr/bin/env bash
# ChemBERTa + GCN late fusion on PPB, seeds 42/43/44.
# Same schedule as ChemBERTa+GIN: 20 epochs, freeze 2, text LR 1e-5,
# graph/head LR 1e-3, batch 16. Graph stack matches standalone GCN (3 x 128, drop 0.25).
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/results"
PIVOT="$ROOT/PPB_pivot"
mkdir -p "$LOGDIR" "$PIVOT"
MASTER="$LOGDIR/multimodal_gcn_ppb.log"

# shellcheck disable=SC1091
source "$ROOT/scripts/activate_dc.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

log "=== START ChemBERTa+GCN PPB seeds 42,43,44 ==="

python scripts/train_multimodal_fusion.py \
  --dataset ppb \
  --seeds 42,43,44 \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-5 \
  --graph-lr 1e-3 \
  --freeze-epochs 2 \
  --graph-encoder gcn \
  --graph-hidden 128 \
  --graph-layers 3 \
  --graph-dropout 0.25 \
  2>&1 | tee -a "$MASTER" \
  | grep -vE "DEPRECATION|normalization for|Skipped loading|No module named|cannot import|Feature removed|OMP:|UNEXPECTED|MISSING|LOAD REPORT|Loading weights" || true

log "=== fusion GCN done; compiling PPB_pivot ==="
python scripts/compile_ppb_pivot.py 2>&1 | tee -a "$MASTER"
log "=== DONE ChemBERTa+GCN PPB ==="
