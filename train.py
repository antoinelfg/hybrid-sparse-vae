"""Hydra-configurable training script for the Hybrid Sparse VAE.

Usage
-----
    python train.py                        # "combat config" defaults
    python train.py n_atoms=32 lr=5e-4     # override via CLI
    python train.py decoder_type=resnet    # deep decoder for real data
    python train.py --cfg job              # print resolved config
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from models.hybrid_vae import HybridSparseVAE
from utils.objectives import compute_hybrid_loss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Hydra structured config — "Combat Config" defaults
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_dir: str = "./data"
    input_channels: int = 1
    input_length: int = 128
    batch_size: int = 64

    # Architecture
    n_atoms: int = 64
    latent_dim: int = 128
    encoder_output_dim: int = 256
    decoder_type: str = "linear"       # "linear" (toy) or "resnet" (real data)
    dict_init: str = "dct"
    normalize_dict: bool = True
    k_min: float = 0.1

    # Training
    lr: float = 1e-3
    epochs: int = 200
    temp_init: float = 1.0
    temp_min: float = 0.1
    temp_anneal_epochs: int = 50

    # KL schedule — "β warm-up" then hold
    #   epoch [1 .. beta_warmup_start]    →  β = 0  (pure reconstruction)
    #   epoch (beta_warmup_start .. beta_warmup_end] → β linearly ↑ to beta_final
    #   epoch > beta_warmup_end           →  β = beta_final
    beta_warmup_start: int = 10        # free reconstruction phase
    beta_warmup_end: int = 50          # end of linear ramp
    beta_gamma_final: float = 0.1      # final β for KL_Gamma
    beta_delta_final: float = 0.1      # final β for KL_Cat

    # Prior — "trou noir" sparse-inducing
    k_0: float = 0.2                   # pulls k toward super-Gaussian regime
    theta_0: float = 1.0

    # Misc
    seed: int = 42
    device: str = "cuda"
    log_every: int = 10
    save_dir: str = "./checkpoints"


# ---------------------------------------------------------------------------
#  Toy data generator — INTERMITTENT sinusoids
# ---------------------------------------------------------------------------

def make_toy_data(
    n_samples: int = 2000,
    length: int = 128,
    n_components: int = 3,
    seed: int = 0,
) -> TensorDataset:
    """Sum of *n_components* sinusoids that switch on/off randomly.

    Each component is active in a random contiguous window of the
    signal (between 30 % and 80 % of the total length).  This forces
    the model to use δ (ternary switch) to handle onset/offset, because
    a continuous decoder cannot predict discontinuities.
    """
    rng = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 2 * torch.pi, length)

    signals = []
    for _ in range(n_samples):
        freqs = torch.randint(1, 20, (n_components,), generator=rng).float()
        amps = torch.rand(n_components, generator=rng) + 0.3   # avoid tiny amps
        phases = torch.rand(n_components, generator=rng) * 2 * torch.pi

        sig = torch.zeros(length)
        for i in range(n_components):
            wave = amps[i] * torch.sin(freqs[i] * t + phases[i])

            # Random on/off window (30 %–80 % of length)
            win_frac = 0.3 + 0.5 * torch.rand(1, generator=rng).item()
            win_len = int(win_frac * length)
            max_start = length - win_len
            start = torch.randint(0, max(max_start, 1), (1,), generator=rng).item()

            mask = torch.zeros(length)
            mask[start : start + win_len] = 1.0
            sig += wave * mask

        signals.append(sig)

    X = torch.stack(signals).unsqueeze(1)  # [N, 1, T]
    return TensorDataset(X)


# ---------------------------------------------------------------------------
#  KL schedule
# ---------------------------------------------------------------------------

def get_beta(epoch: int, cfg: TrainConfig) -> float:
    """β warm-up schedule: 0 → β_final with linear ramp."""
    if epoch <= cfg.beta_warmup_start:
        return 0.0
    if epoch >= cfg.beta_warmup_end:
        return 1.0   # multiplied by beta_*_final in the loss call
    frac = (epoch - cfg.beta_warmup_start) / max(
        cfg.beta_warmup_end - cfg.beta_warmup_start, 1
    )
    return frac


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---- Data ----------------------------------------------------------
    dataset = make_toy_data(
        n_samples=2000,
        length=cfg.input_length,
        n_components=3,
        seed=cfg.seed,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # ---- Model ---------------------------------------------------------
    model = HybridSparseVAE(
        input_channels=cfg.input_channels,
        input_length=cfg.input_length,
        encoder_output_dim=cfg.encoder_output_dim,
        n_atoms=cfg.n_atoms,
        latent_dim=cfg.latent_dim,
        decoder_type=cfg.decoder_type,
        dict_init=cfg.dict_init,
        normalize_dict=cfg.normalize_dict,
        k_min=cfg.k_min,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Temperature schedule (Gumbel-Softmax)
    def get_temp(epoch: int) -> float:
        if epoch >= cfg.temp_anneal_epochs:
            return cfg.temp_min
        frac = epoch / max(cfg.temp_anneal_epochs, 1)
        return cfg.temp_init + (cfg.temp_min - cfg.temp_init) * frac

    # ---- Training loop -------------------------------------------------
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        temp = get_temp(epoch)
        beta_frac = get_beta(epoch, cfg)

        # Effective betas for this epoch
        eff_beta_gamma = beta_frac * cfg.beta_gamma_final
        eff_beta_delta = beta_frac * cfg.beta_delta_final

        epoch_loss = 0.0
        epoch_metrics: dict[str, float] = {
            "recon": 0.0, "kl_gamma": 0.0, "kl_delta": 0.0
        }

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_recon, info = model(batch_x, temp=temp)

            loss, metrics = compute_hybrid_loss(
                x=batch_x,
                x_recon=x_recon,
                params=info,
                k_0=cfg.k_0,
                theta_0=cfg.theta_0,
                beta_gamma=eff_beta_gamma,
                beta_delta=eff_beta_delta,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key].item()

        n_batches = len(loader)
        if epoch % cfg.log_every == 0 or epoch == 1:
            avg = {k: v / n_batches for k, v in epoch_metrics.items()}
            k_mean = info["k"].mean().item()
            k_min_val = info["k"].min().item()
            sparsity = (info["delta"] == 0).float().mean().item()
            log.info(
                f"Epoch {epoch:4d} | loss {epoch_loss/n_batches:.4f} | "
                f"recon {avg['recon']:.4f} | kl_γ {avg['kl_gamma']:.4f} | "
                f"kl_δ {avg['kl_delta']:.4f} | "
                f"k̄={k_mean:.3f}  k_min={k_min_val:.3f}  "
                f"δ₀={sparsity:.2%}  "
                f"β={eff_beta_gamma:.4f}  τ={temp:.3f}"
            )

    # ---- Save ----------------------------------------------------------
    ckpt_path = Path(cfg.save_dir) / "hybrid_vae_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    log.info(f"Saved checkpoint → {ckpt_path}")


# ---------------------------------------------------------------------------
#  Hydra entry-point
# ---------------------------------------------------------------------------

cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)


@hydra.main(config_path=None, config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_cfg: TrainConfig = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    train(train_cfg)


if __name__ == "__main__":
    main()
