#!/usr/bin/env bash
# Robust GAT multi-seed runner (one job at a time, unbuffered logs, continue on failure)
set -u
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/results/gat_robust_rerun.log"
mkdir -p "$ROOT/results"

# shellcheck disable=SC1091
source "$ROOT/scripts/activate_dc.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== GAT robust re-run START ==="
log "Why previous run failed: only PPB seed=42 completed (~42 min)."
log "Seeds 43/44 and hopv/clearance never started — process almost certainly"
log "died after seed 42 (likely MPS/RAM pressure). stdout was pipe-buffered so"
log "the crash left almost nothing in pyg_gat_multiseed.log."

DATASETS=(ppb hopv clearance)
SEEDS=(42 43 44)

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # Skip if full e100 metrics already exist for this seed
    metrics="$ROOT/results/${ds}/PyG_GAT_h128_e100/seed${seed}/metrics.json"
    if [[ -f "$metrics" ]]; then
      log "SKIP ${ds} seed=${seed} (already have ${metrics})"
      continue
    fi
    log "RUN  ${ds} seed=${seed} ..."
    # Isolate each run in its own python process so one crash cannot kill the loop
    if python -u scripts/train_pyg_gnn.py \
        --arch gat \
        --dataset "$ds" \
        --seeds "$seed" \
        --epochs 100 \
        --patience 25 \
        --hidden 128 \
        --batch-size 16 \
        2>&1 | tee -a "$LOG" | grep -E "\[${ds}|ERROR|Traceback|early stop|test R2|n_feat" || true
    then
      :
    fi
    # force free memory between runs
    python -u - <<'PY' 2>/dev/null || true
import gc
try:
    import torch
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
except Exception:
    pass
PY
    if [[ -f "$metrics" ]]; then
      log "OK   ${ds} seed=${seed}"
    else
      log "FAIL ${ds} seed=${seed} — no metrics.json written"
    fi
  done
done

log "=== Summarize + figures ==="
python -u scripts/summarize_leaderboard.py 2>&1 | tee -a "$LOG" || true
python -u scripts/plot_publication_figures.py 2>&1 | tee -a "$LOG" || true
log "=== GAT robust re-run DONE ==="
