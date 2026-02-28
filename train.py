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
import math
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
    n_atoms: int = 128                  # overcomplete: 2x latent_dim
    latent_dim: int = 64
    encoder_output_dim: int = 256
    encoder_type: str = "linear"       # "linear" (toy) or "resnet" (real data)
    decoder_type: str = "linear"       # "linear" (toy) or "resnet" (real data)
    dict_init: str = "dct"
    normalize_dict: bool = True
    k_min: float = 0.1

    # Dictionary learning
    freeze_dict_until: int = 0          # freeze dict weights until this epoch (0=always learn)
    dict_lr_mult: float = 0.1           # dict LR = lr * dict_lr_mult (slower adaptation)

    # Training
    lr: float = 3e-4
    epochs: int = 3000
    temp_init: float = 1.0
    temp_min: float = 0.05
    temp_anneal_epochs: int = 50

    # 4-Phase "Champion" schedule
    #   P1 [1..phase1_end]              : soft, β=0    (sculpt topology)
    #   P2 (phase1..phase2_end]         : stoch, β=0   (immunize to noise)
    #   P3 (phase2..phase3_end]         : stoch, β ramp (sparsify)
    #   P4 (phase3..epochs]             : stoch, β=max  (stationary convergence)
    phase1_end: int = 400
    phase2_end: int = 500
    phase3_end: int = 1000
    beta_gamma_final: float = 0.005     # flexible magnitude
    beta_delta_final: float = 0.1       # strict structure

    # Prior
    k_0: float = 0.3                    # low prior → less KL tax
    theta_0: float = 1.0

    # Delta prior: [P(-1), P(0), P(+1)] — relaxed to allow more active atoms
    delta_prior: str = "0.15,0.70,0.15"  # serialized as string for Hydra

    # Misc
    seed: int = 42
    device: str = "cuda"
    log_every: int = 10
    save_dir: str = "./checkpoints"


# ---------------------------------------------------------------------------
#  Toy data generator — continuous sinusoids (simple first)
# ---------------------------------------------------------------------------

def make_toy_data(
    n_samples: int = 2000,
    length: int = 128,
    n_components: int = 3,
    seed: int = 0,
) -> TensorDataset:
    """Sum of *n_components* continuous sinusoids.

    Simple baseline: get this working first (k should drop below 1),
    then graduate to intermittent/gated signals.
    """
    rng = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 2 * torch.pi, length)

    signals = []
    for _ in range(n_samples):
        freqs = torch.randint(1, 20, (n_components,), generator=rng).float()
        amps = torch.rand(n_components, generator=rng) + 0.3
        phases = torch.rand(n_components, generator=rng) * 2 * torch.pi
        sig = sum(a * torch.sin(f * t + p) for a, f, p in zip(amps, freqs, phases))
        signals.append(sig)

    X = torch.stack(signals).unsqueeze(1)  # [N, 1, T]

    # --- Normalize to ~[-1, 1] so MSE is commensurate with KL ---
    # Max theoretical amplitude ≈ 3 × 1.3 = 3.9.  Divide by 4.
    X = X / 4.0

    return TensorDataset(X)


# ---------------------------------------------------------------------------
#  KL schedule
# ---------------------------------------------------------------------------

def get_phase(epoch: int, cfg: TrainConfig) -> tuple[str, float]:
    """4-phase schedule: soft → stoch → ramp → stationary."""
    if epoch <= cfg.phase1_end:
        return "soft", 0.0
    if epoch <= cfg.phase2_end:
        return "stochastic", 0.0
    if epoch <= cfg.phase3_end:
        # Phase 3: KL ramp 0 → 1.0
        ramp_len = max(cfg.phase3_end - cfg.phase2_end, 1)
        frac = (epoch - cfg.phase2_end) / ramp_len
        return "stochastic", min(frac, 1.0)
    # Phase 4: stationary at β=final
    return "stochastic", 1.0


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
        encoder_type=cfg.encoder_type,
        encoder_output_dim=cfg.encoder_output_dim,
        n_atoms=cfg.n_atoms,
        latent_dim=cfg.latent_dim,
        decoder_type=cfg.decoder_type,
        dict_init=cfg.dict_init,
        normalize_dict=cfg.normalize_dict,
        k_min=cfg.k_min,
    ).to(device)

    # Separate param groups: dict learns slower
    dict_params = list(model.latent.dictionary.parameters())
    dict_ids = {id(p) for p in dict_params}
    other_params = [p for p in model.parameters() if id(p) not in dict_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": cfg.lr},
        {"params": dict_params, "lr": cfg.lr * cfg.dict_lr_mult, "name": "dict"},
    ])

    # Store initial dict for drift measurement
    dict_init_snapshot = model.latent.dictionary.get_atoms().clone()

    # LR schedule: constant through P1-P3, cosine decay in P4
    def get_lr(epoch: int) -> float:
        if epoch <= cfg.phase3_end:
            return cfg.lr
        # Cosine decay from lr → lr/10 during Phase 4
        frac = (epoch - cfg.phase3_end) / max(cfg.epochs - cfg.phase3_end, 1)
        lr_min = cfg.lr / 10.0
        return lr_min + 0.5 * (cfg.lr - lr_min) * (1 + math.cos(math.pi * frac))

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
        sampling_mode, beta_frac = get_phase(epoch, cfg)

        # Update LR (cosine decay in P4)
        current_lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            if pg.get('name') == 'dict':
                pg['lr'] = current_lr * cfg.dict_lr_mult
            else:
                pg['lr'] = current_lr

        # Freeze/unfreeze dictionary
        dict_frozen = epoch <= cfg.freeze_dict_until
        for p in dict_params:
            p.requires_grad = not dict_frozen

        # Effective betas for this epoch
        eff_beta_gamma = beta_frac * cfg.beta_gamma_final
        eff_beta_delta = beta_frac * cfg.beta_delta_final

        # Phase label for logging
        if epoch <= cfg.phase1_end:
            phase_label = "P1:soft"
        elif epoch <= cfg.phase2_end:
            phase_label = "P2:stoch"
        elif epoch <= cfg.phase3_end:
            phase_label = "P3:ramp"
        else:
            phase_label = "P4:conv"

        epoch_loss = 0.0
        epoch_metrics: dict[str, float] = {
            "recon": 0.0, "kl_gamma": 0.0, "kl_delta": 0.0
        }

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_recon, info = model(batch_x, temp=temp, sampling=sampling_mode)

            # Parse delta prior
            dp = [float(v) for v in cfg.delta_prior.split(",")]
            delta_prior_t = torch.tensor(dp, device=device)

            loss, metrics = compute_hybrid_loss(
                x=batch_x,
                x_recon=x_recon,
                params=info,
                k_0=cfg.k_0,
                theta_0=cfg.theta_0,
                prior_probs=delta_prior_t,
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
            k_all = info["k"]
            delta = info["delta"]
            active_mask = (delta != 0)
            k_mean = k_all.mean().item()
            k_active = k_all[active_mask].mean().item() if active_mask.any() else 0.0
            n_active = active_mask.float().sum(-1).mean().item()
            sparsity = (delta == 0).float().mean().item()
            # Dictionary drift from initialization
            current_atoms = model.latent.dictionary.get_atoms()
            dict_drift = 1.0 - F.cosine_similarity(
                current_atoms.flatten().unsqueeze(0),
                dict_init_snapshot.flatten().unsqueeze(0),
            ).item()
            log.info(
                f"Epoch {epoch:4d} [{phase_label}] | "
                f"loss {epoch_loss/n_batches:.4f} | "
                f"recon {avg['recon']:.4f} | kl_γ {avg['kl_gamma']:.4f} | "
                f"kl_δ {avg['kl_delta']:.4f} | "
                f"k̄={k_mean:.3f}  k_act={k_active:.3f}  "
                f"n_act={n_active:.1f}/{cfg.n_atoms}  "
                f"δ₀={sparsity:.2%}  "
                f"β={eff_beta_gamma:.4f}  τ={temp:.3f}  "
                f"Δdict={dict_drift:.4f}"
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
