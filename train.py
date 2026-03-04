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
    dataset: str = "sinusoid"           # "sinusoid", "mnist", "audio"
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
    dict_init: str = "random"
    normalize_dict: bool = True
    k_min: float = 0.1

    # Dictionary learning
    freeze_dict_until: int = 0          # freeze dict weights until this epoch (0=always learn)
    dict_lr_mult: float = 0.1           # dict LR = lr * dict_lr_mult (slower adaptation)
    
    # Warmup strategies (dict stability)
    dict_warmup_epochs: int = 0         # freeze dict for first N epochs
    dict_lr_warmup: bool = False        # linear LR warmup for dict params over first 200 epochs
    gradient_clip_dict: float = 1.0     # separate grad clip for dict
    
    # Ablations
    magnitude_dist: str = "gamma"       # "gamma" or "gaussian"
    structure_mode: str = "ternary"     # "ternary" or "binary"
    spectrogram_enhancements: bool = True # Force non-negativity, instance norm, higher sparsity for spectrograms

    # WandB
    use_wandb: bool = False
    wandb_project: str = "hybrid-sparse-vae"
    wandb_entity: str = ""
    wandb_run_name: str = ""

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
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity if cfg.wandb_entity else None,
                name=cfg.wandb_run_name if cfg.wandb_run_name else None,
                config=vars(cfg) if not isinstance(cfg, DictConfig) else OmegaConf.to_container(cfg, resolve=True),
            )
        except ImportError:
            log.warning("wandb not installed. Run `pip install wandb` to use it. Disabling wandb.")
            cfg.use_wandb = False

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---- Data ----------------------------------------------------------
    dataset_name = cfg.dataset.lower()
    
    if dataset_name == "sinusoid":
        dataset = make_toy_data(
            n_samples=2000,
            length=cfg.input_length,
            n_components=3,
            seed=cfg.seed,
        )
    elif dataset_name == "mnist":
        from data.datasets import get_mnist_dataset
        dataset = get_mnist_dataset(data_dir=cfg.data_dir, flatten=True)
    elif dataset_name == "audio":
        from data.datasets import get_audio_spectrogram_dataset
        # For our 1D convolutions, we treat n_mels as channels and time_steps as length.
        # But wait, our get_audio_spectrogram_dataset returns [N, C, T] where C=n_mels.
        # So we MUST adjust input_channels and input_length if we plan to use audio.
        dataset = get_audio_spectrogram_dataset(
            data_dir=cfg.data_dir,
            n_samples=2000,
            n_mels=cfg.input_channels,
            time_steps=cfg.input_length,
            use_instance_norm=cfg.spectrogram_enhancements
        )
    elif dataset_name == "fsdd":
        from data.datasets import get_fsdd_dataset
        # For our purposes we assume defaults n_fft=256, hop_length=128, max_frames=64
        # to match input_channels=1, input_length=8256
        dataset = get_fsdd_dataset(
            data_dir=f"{cfg.data_dir}/fsdd",
            use_instance_norm=cfg.spectrogram_enhancements
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
        
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # ---- Auto-detect shape to avoid magic constants --------------------
    if len(dataset) > 0:
        sample_x = dataset[0][0]  # TensorDataset returns tuples
        if sample_x.dim() == 2:
            cfg.input_channels = sample_x.shape[0]
            cfg.input_length = sample_x.shape[1]
        elif sample_x.dim() == 1:
            cfg.input_channels = 1
            cfg.input_length = sample_x.shape[0]

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
        magnitude_dist=cfg.magnitude_dist,
        structure_mode=cfg.structure_mode,
    ).to(device)

    # Separate param groups: dict learns slower
    if getattr(model.latent, "temporal_mode", False):
        dict_params = list(model.decoder.parameters())
    else:
        dict_params = list(model.latent.dictionary.parameters())
    
    dict_ids = {id(p) for p in dict_params}
    other_params = [p for p in model.parameters() if id(p) not in dict_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": cfg.lr},
        {"params": dict_params, "lr": cfg.lr * cfg.dict_lr_mult, "name": "dict"},
    ])

    # Store initial dict for drift measurement
    dict_init_snapshot = model.get_dict_atoms().clone()

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
                if cfg.dict_lr_warmup and epoch <= 200:
                    warmup_factor = epoch / 200.0
                    pg['lr'] = current_lr * cfg.dict_lr_mult * warmup_factor
                else:
                    pg['lr'] = current_lr * cfg.dict_lr_mult
            else:
                pg['lr'] = current_lr

        # Freeze/unfreeze dictionary (combine explicit freeze with warmup)
        dict_frozen = epoch <= max(cfg.freeze_dict_until, cfg.dict_warmup_epochs)
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

        for batch in loader:
            batch_x = batch[0]
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_recon, info = model(batch_x, temp=temp, sampling=sampling_mode)

            # Parse delta prior
            dp = [float(v) for v in cfg.delta_prior.split(",")]
            
            # Auto-adapt prior to structure mode
            if cfg.structure_mode == "binary" and len(dp) == 3:
                dp = [dp[1], dp[0] + dp[2]]  # [P(0), P(-1) + P(+1)]
            elif cfg.structure_mode == "ternary" and len(dp) == 2:
                dp = [dp[1]/2, dp[0], dp[1]/2]  # Split P(active) to -1 and +1
                
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
                magnitude_dist=cfg.magnitude_dist,
            )

            loss.backward()
            
            # Separate gradient clipping for dictionary stability
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=5.0)
            if not dict_frozen:
                torch.nn.utils.clip_grad_norm_(dict_params, max_norm=cfg.gradient_clip_dict)
                
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
            # Moyenne d'atomes actifs SIMULTANÉMENT (par trame temporelle)
            # Somme sur la dimension des atomes (dim=1), moyenne sur le Batch et le Temps
            n_active_per_frame = active_mask.float().sum(dim=1).mean().item()
            
            # Nombre d'atomes distincts utilisés AU MOINS UNE FOIS dans tout le fichier audio
            if active_mask.dim() == 3:
                n_active_total = active_mask.any(dim=2).float().sum(dim=1).mean().item()
            else:
                n_active_total = n_active_per_frame

            sparsity = (delta == 0).float().mean().item()
            # Dictionary drift from initialization
            current_atoms = model.get_dict_atoms()
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
                f"n_act_frame={n_active_per_frame:.1f}  n_act_total={n_active_total:.1f}/{cfg.n_atoms}  "
                f"δ₀={sparsity:.2%}  "
                f"β={eff_beta_gamma:.4f}  τ={temp:.3f}  "
                f"Δdict={dict_drift:.4f}"
            )
            
            if cfg.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "loss": epoch_loss / n_batches,
                    "recon": avg["recon"],
                    "kl_gamma": avg["kl_gamma"],
                    "kl_delta": avg["kl_delta"],
                    "k_mean": k_mean,
                    "k_active": k_active,
                    "n_active": n_active,
                    "sparsity": sparsity,
                    "dict_drift": dict_drift,
                    "phase_beta_gamma": eff_beta_gamma,
                    "phase_beta_delta": eff_beta_delta,
                    "temp": temp,
                })

    # ---- Save ----------------------------------------------------------
    ckpt_path = Path(cfg.save_dir) / "hybrid_vae_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    log.info(f"Saved checkpoint → {ckpt_path}")
    
    if cfg.use_wandb:
        import wandb
        wandb.save(str(ckpt_path))
        wandb.finish()


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
