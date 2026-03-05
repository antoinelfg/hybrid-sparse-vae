import torch
import torch.nn.functional as F
import numpy as np
from models.hybrid_vae import HybridSparseVAE
from data.datasets import get_dataloader

def run_diagnostic(checkpoint_path, denoise=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Auto-detect config from checkpoint or use common defaults
    # For final_push: motif_width=32, n_atoms=128
    model = HybridSparseVAE(
        input_channels=129, input_length=64,
        encoder_type='resnet', decoder_type='convnmf',
        n_atoms=128, latent_dim=64,
        motif_width=32, decoder_stride=16,
        k_max=0.8
    ).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    dataloader = get_dataloader("fsdd", batch_size=64, train=False, denoise=denoise)
    
    total_signal_mse = 0
    total_silence_mse = 0
    total_raw_mse = 0
    total_k_bar = 0
    total_n_act = 0
    n_batches = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_recon, info = model(x, temp=0.05, sampling='deterministic')
            
            mask = (x > 1e-4).float()
            
            # MSE on signal
            sig_err = ((x_recon - x) * mask).pow(2).sum() / (mask.sum() + 1e-6)
            # Energy in silence (hallucination)
            sil_err = (x_recon * (1 - mask)).pow(2).sum() / ((1 - mask).sum() + 1e-6)
            # Raw MSE
            raw_mse = F.mse_loss(x_recon, x)

            total_signal_mse += sig_err.item()
            total_silence_mse += sil_err.item()
            total_raw_mse += raw_mse.item()
            total_k_bar += info['k'].mean().item()
            total_n_act += (info['delta'] > 0.5).float().sum(dim=1).mean().item()
            n_batches += 1

    print(f"--- Diagnostic: {checkpoint_path} ---")
    print(f"Signal MSE:      {total_signal_mse / n_batches:.6f}")
    print(f"Silence Energy:  {total_silence_mse / n_batches:.6f}")
    print(f"Raw MSE:         {total_raw_mse / n_batches:.6f}")
    print(f"Avg k̄:           {total_k_bar / n_batches:.4f}")
    print(f"Avg Act/Sample:  {total_n_act / n_batches:.2f} (active frames total)")

if __name__ == "__main__":
    run_diagnostic("checkpoints/fsdd_final_push/hybrid_vae_final.pt")
