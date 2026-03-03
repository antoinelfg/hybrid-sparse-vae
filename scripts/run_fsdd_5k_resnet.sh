#!/bin/bash
#SBATCH --job-name=fsdd_resnet
#SBATCH --output=fsdd_resnet_%j.out
#SBATCH --error=fsdd_resnet_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

python train.py \
    dataset="fsdd" \
    input_length=8256 \
    decoder_type="linear" \
    encoder_type="resnet" \
    n_atoms=64 \
    latent_dim=32 \
    dict_init="random" \
    k_min=0.01 \
    magnitude_dist="gamma" \
    structure_mode="binary" \
    epochs=5000 \
    phase1_end=1500 \
    phase2_end=2500 \
    phase3_end=4000 \
    dict_warmup_epochs=0 \
    use_wandb=False \
    save_dir="./checkpoints/fsdd_5k_resnet"
