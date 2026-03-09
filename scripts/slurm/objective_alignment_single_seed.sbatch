#!/bin/bash
#SBATCH --job-name=fpolar_obj_1seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=objective_alignment_single_seed_%j.out
#SBATCH --error=objective_alignment_single_seed_%j.err

set -euo pipefail

REPO_ROOT="/home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae"
cd "$REPO_ROOT"
export PYTHONPATH=.

SEED_SPEC="${SEED_SPEC:-${SEED:-456}}"
EPOCHS="${EPOCHS:-1000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/regime_study_objective}"
CONDITIONS_SPEC="${CONDITIONS_SPEC:-HSVAE-fully-polar-lista-softprior-energy-div4-p20-bp020-dictinit-identity-dictlr-010-gammakl-tilde-pcons-000:HSVAE-fully-polar-lista-softprior-energy-div4-p20-bp020-dictinit-identity-dictlr-010-gammakl-final-pcons-000:HSVAE-fully-polar-lista-softprior-energy-div4-p20-bp020-dictinit-identity-dictlr-010-gammakl-tilde-pcons-010:HSVAE-fully-polar-lista-softprior-energy-div4-p20-bp020-dictinit-identity-dictlr-010-gammakl-tilde-pcons-020:HSVAE-fully-polar-lista-softprior-energy-div4-p20-bp020-dictinit-identity-dictlr-010-gammakl-tilde-pcons-050}"
SEEDS="${SEED_SPEC//:/,}"
CONDITIONS_CSV="${CONDITIONS_SPEC//:/,}"

echo "============================================"
echo " Fully Polar objective sweep (single seed)"
echo " Node: $(hostname)"
echo " Seeds: $SEEDS"
echo " Epochs: $EPOCHS"
echo " Device: $DEVICE"
echo " Output root: $OUTPUT_ROOT"
echo " Conditions: $CONDITIONS_CSV"
echo " Started at: $(date)"
echo "============================================"

python scripts/run_regime_multiseed.py \
  --conditions "$CONDITIONS_CSV" \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --skip-existing

echo "============================================"
echo " Finished at: $(date)"
echo "============================================"
