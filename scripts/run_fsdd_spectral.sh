#!/bin/bash
# Train Hybrid Sparse VAE on the Free Spoken Digit Dataset (FSDD) Spectrograms
# 
# We use a purely linear dictionary/decoder to prove that the model 
# learns a Deep Non-negative Matrix Factorization (NMF) of the harmonic structures.
# Free Spoken Digit Dataset audio -> STFT Spectrograms -> Flattened.
#
# Constraints:
# n_fft=256, hop_length=128, max_frames=64
# -> Freq bins = (256 // 2 + 1) = 129
# -> Time frames = 64
# -> Input Length = 129 * 64 = 8256

set -e

# Run training
python train.py \
    dataset="fsdd" \
    input_channels=1 \
    decoder_type="linear_positive" \
    encoder_type="mlp" \
    n_atoms=64 \
    latent_dim=32 \
    dict_init="random_positive" \
    k_min=0.01 \
    magnitude_dist="gamma" \
    structure_mode="ternary" \
    dict_warmup_epochs=100 \
    freeze_dict_until=50 \
    dict_lr_warmup=True \
    epochs=1000 \
    phase1_end=200 \
    phase2_end=300 \
    phase3_end=600 \
    wandb_run_name="fsdd_spectral_linear" \
    use_wandb=False \
    save_dir="./checkpoints/fsdd_spectral"

echo "Training complete. Run vis_spectrogram_atoms.py to see the learned harmonic atoms."
