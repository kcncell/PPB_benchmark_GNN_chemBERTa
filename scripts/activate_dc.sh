# Source from run_*.sh after ROOT is set.
# Activates the conda env `dc` without a machine-specific Miniconda path.
if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "${CONDA_PREFIX}")" = "dc" ]; then
  return 0 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "${HOME}/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/opt/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda not found. Install Miniconda and create env dc (see README.md)." >&2
  exit 1
fi
conda activate dc
