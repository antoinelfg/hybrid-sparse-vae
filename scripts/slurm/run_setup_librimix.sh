#!/bin/bash
#SBATCH --job-name=setup_librimix
#SBATCH --output=setup_librimix_%j.out
#SBATCH --error=setup_librimix_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -eo pipefail

# Some cluster bashrc files use unbound vars; avoid failing during source.
if [[ -f ~/.bashrc ]]; then
  set +u
  source ~/.bashrc
  set -u
fi

if command -v conda >/dev/null 2>&1; then
  conda activate base || true
fi

# Force conda C++ runtime (fixes scipy ImportError on older system libstdc++).
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# In SLURM, BASH_SOURCE may point to a copy under /var/spool/slurm.
# Prefer the original submission directory when available.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$SUBMIT_DIR/scripts/setup_librimix.sh" ]]; then
  REPO_ROOT="$SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

# You can override these at submission time, e.g.:
#   sbatch --export=ALL,STORAGE_DIR=/scratch/.../data,SKIP_PIP_INSTALL=1 scripts/slurm/run_setup_librimix.sh
STORAGE_DIR="${STORAGE_DIR:-$REPO_ROOT/data}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
ENABLE_WHAM_AUGMENT="${ENABLE_WHAM_AUGMENT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========================================="
echo "Libri2Mix Setup (SLURM)"
echo "Date: $(date)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Storage dir: $STORAGE_DIR"
echo "SKIP_PIP_INSTALL=$SKIP_PIP_INSTALL"
echo "ENABLE_WHAM_AUGMENT=$ENABLE_WHAM_AUGMENT"
echo "========================================="

mkdir -p "$STORAGE_DIR"

env \
  SKIP_PIP_INSTALL="$SKIP_PIP_INSTALL" \
  ENABLE_WHAM_AUGMENT="$ENABLE_WHAM_AUGMENT" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/setup_librimix.sh "$STORAGE_DIR" \
  2>&1 | tee "$SUBMIT_DIR/setup_librimix_slurm_${SLURM_JOB_ID:-manual}.log"

echo "========================================="
echo "Libri2Mix setup finished: $(date)"
echo "Expected output root: $STORAGE_DIR/Libri2Mix"
echo "========================================="
