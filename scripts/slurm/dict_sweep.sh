#!/bin/bash
#SBATCH --job-name=dict_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=dict_sweep_%j.out
#SBATCH --error=dict_sweep_%j.err

# =============================================================================
#  Dictionary Learning Sweep
#
#  Tests whether learning the dictionary (vs fixed DCT) improves recon.
#  All configs use champion v2 settings + 3000 epochs.
# =============================================================================

cd /home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae
export PYTHONPATH=.

RESULTS_DIR="./results/dict_sweep_$(date +%Y%m%d_%H%M)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Dictionary Learning Sweep — $(date)"
echo "  Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

run_config() {
    local name="$1"
    shift
    local run_dir="$RESULTS_DIR/$name"
    mkdir -p "$run_dir"
    echo ""
    echo ">>> [$name] Starting at $(date +%H:%M:%S)"
    python train.py hydra.run.dir="$run_dir" "$@" 2>&1 | tee "$run_dir/train.log"
    echo ">>> [$name] Finished at $(date +%H:%M:%S)"
}

# === Config 1: Baseline — DCT always learnable (same as v2 but explicit) ===
run_config "dct_learn_always" \
    epochs=3000 dict_init=dct freeze_dict_until=0 dict_lr_mult=0.1

# === Config 2: DCT init, freeze until end of Phase 2 ===
# (topology first, then co-adapt dict)
run_config "dct_freeze_p12" \
    epochs=3000 dict_init=dct freeze_dict_until=500 dict_lr_mult=0.1

# === Config 3: DCT init, freeze until end of Phase 3 ===
# (let KL set sparsity first, then tune dict for convergence)
run_config "dct_freeze_p123" \
    epochs=3000 dict_init=dct freeze_dict_until=1000 dict_lr_mult=0.1

# === Config 4: Random init, always learnable ===
# (can the encoder co-adapt to arbitrary atoms?)
run_config "rand_learn_always" \
    epochs=3000 dict_init=random freeze_dict_until=0 dict_lr_mult=0.1

# === Config 5: Random init, higher dict LR ===
run_config "rand_lr_0.3" \
    epochs=3000 dict_init=random freeze_dict_until=0 dict_lr_mult=0.3

# === Config 6: DCT init, higher dict LR ===
run_config "dct_lr_0.3" \
    epochs=3000 dict_init=dct freeze_dict_until=0 dict_lr_mult=0.3

# === Config 7: DCT frozen (control — same as previous champion v2) ===
run_config "dct_frozen" \
    epochs=3000 dict_init=dct freeze_dict_until=9999 dict_lr_mult=0.0

# === Config 8: Random init, freeze P1, then learn with high LR ===
run_config "rand_freeze_p1_lr0.5" \
    epochs=3000 dict_init=random freeze_dict_until=400 dict_lr_mult=0.5

# Summary
echo ""
echo "=== FINAL EPOCH SUMMARY ==="
for dir in "$RESULTS_DIR"/*/; do
    name=$(basename "$dir")
    last_line=$(grep "Epoch" "$dir/train.log" 2>/dev/null | tail -1 || echo "NO DATA")
    echo "[$name] $last_line"
done
