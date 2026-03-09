"""Hydra-configurable training script for the Hybrid Sparse VAE.

Usage
-----
    python train.py                        # "combat config" defaults
    python train.py n_atoms=32 lr=5e-4     # override via CLI
    python train.py decoder_type=resnet    # deep decoder for real data
    python train.py --cfg job              # print resolved config
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from models.hybrid_vae import HybridSparseVAE
from utils.objectives import compute_fully_polar_loss, compute_hybrid_loss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Hydra structured config — "Combat Config" defaults
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    dataset: str = "sinusoid"           # "sinusoid", "mnist", "audio", "fsdd", "librimix"
    data_dir: str = "./data"
    input_channels: int = 1
    input_length: int = 128
    max_frames: int = 64                # Temporal dimension for spectrograms
    batch_size: int = 64
    librimix_split: str = "train-100"
    librimix_mix_type: str = "min"
    librimix_mixture_dirname: str = "mix_clean"
    librimix_sample_rate: int = 8000
    librimix_n_fft: int = 512
    librimix_hop_length: int = 128
    librimix_win_length: int = 512
    librimix_root_dir: str = ""         # Optional explicit root. If empty: {data_dir}/Libri2Mix
    librimix_crop_mode: str = "center"

    # Architecture
    n_atoms: int = 128                  # overcomplete: 2x latent_dim
    latent_dim: int = 64
    encoder_output_dim: int = 256
    encoder_type: str = "linear"       # "linear", "lista", "polar_lista", "fully_polar_lista", or "resnet"
    decoder_type: str = "linear"       # "linear" (toy) or "resnet" (real data)
    dict_init: str = "random"
    normalize_dict: bool = True
    k_min: float = 0.1
    k_max: float = 1e9                   # Optional Gamma shape ceiling (1e9 = unconstrained)
    motif_width: int = 16                # ConvNMF atom width (use 64 for Overlap-Add)
    decoder_stride: int = -1             # ConvNMF stride (-1 = auto: max_frames // 4)
    match_encoder_decoder_stride: bool = False
    lista_iterations: int = 5
    lista_threshold_init: float = 0.1
    delta_head_mode: str = "shared"    # "shared" or "l2norm"
    polar_encoder: bool = False
    fully_polar_encoder: bool = False
    shape_norm: str = "l2_global"
    gain_feature: str = "log_l2"
    gamma_scale_injection: str = "multiply_input_norm"
    shape_detach_to_gamma: bool = True
    delta_factorization: str = "ternary_direct"  # "ternary_direct" or "presence_sign"
    presence_estimator: str = "gumbel_binary"    # "gumbel_binary", "sparsemax_binary", "entmax15_binary"
    sign_estimator: str = "gumbel_binary"
    presence_alpha: float = 1.5
    tau_presence_eval: float = 0.5
    sign_tau_eval: float = 0.5
    presence_head_bias_init: float = 0.0
    sign_head_bias_init: float = 0.0
    gumbel_epsilon: float = 0.05
    sinusoid_gain_distribution: str = "none"
    sinusoid_gain_min: float = 1.0
    sinusoid_gain_max: float = 1.0
    sinusoid_normalize_divisor: float = 4.0

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
    denoise: bool = False                  # Wiener-style spectral floor subtraction before log
    masked_recon: bool = False             # Use masked MSE (signal-only loss) — pair with denoise=True
    lambda_silence: float = 0.05          # Silence suppression weight (initial / P1-P2)
    lambda_silence_final: float = 0.05    # Silence suppression weight at end of P4 (progressive pruning)
    lambda_recon_l1: float = 0.0          # L1 noise-floor penalty on x_recon (drives output toward 0)
    lambda_atom_coherence: float = 0.0    # Off-diagonal Gram penalty on effective atoms

    # WandB
    use_wandb: bool = False
    wandb_project: str = "hybrid-sparse-vae"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_bss_eval_every: int = 10
    wandb_bss_media_every: int = 25
    wandb_bss_max_eval: int = 8
    wandb_bss_split: str = "dev"
    wandb_bss_alpha: float = 0.5
    wandb_bss_h_representation: str = "B_abs"
    wandb_bss_mask_power: float = 2.0
    wandb_bss_cluster_method: str = "fiedler_median"

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
    #   P5 [soft_finetune_start..epochs]: soft, β=max   (optional train/eval alignment)
    phase1_end: int = 400
    phase2_end: int = 500
    phase3_end: int = 1000
    soft_finetune_start: int = 0      # Optional late soft phase (0 = disabled)
    beta_gamma_final: float = 0.005     # flexible magnitude
    beta_delta_final: float = 0.1       # strict structure
    beta_presence_final: float = 0.1
    beta_sign_final: float = 0.02
    lambda_presence_consistency_final: float = 0.0
    kl_normalization: str = "site"      # "batch" or "site"
    gamma_kl_target: str = "theta_tilde"  # "theta_tilde" or "theta_final"
    presence_consistency_target: str = "argmax"

    # Prior
    k_0: float = 0.3                    # low prior → less KL tax
    theta_0: float = 1.0

    # Delta prior: [P(-1), P(0), P(+1)] — relaxed to allow more active atoms
    delta_prior: str = "0.15,0.70,0.15"  # serialized as string for Hydra
    presence_prior: str = "0.90,0.10"
    sign_prior: str = "0.50,0.50"

    # Misc
    seed: int = 42
    device: str = "cuda"
    log_every: int = 10
    save_dir: str = "./checkpoints"
    sinusoid_sparse_monitor_every: int = 0
    sinusoid_sparse_monitor_samples: int = 512
    sinusoid_sparse_monitor_components: int = 3
    sinusoid_sparse_monitor_seed: int = 0
    sinusoid_sparse_monitor_max_frequency: int = 19


# ---------------------------------------------------------------------------
#  Toy data generator — continuous sinusoids (simple first)
# ---------------------------------------------------------------------------

def generate_toy_sinusoid_tensors(
    n_samples: int = 2000,
    length: int = 128,
    n_components: int = 3,
    seed: int = 0,
    gain_distribution: str = "none",
    gain_min: float = 1.0,
    gain_max: float = 1.0,
    normalize_divisor: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate toy sinusoid tensors plus exact metadata.

    Returns
    -------
    X : Tensor [N, 1, T]
        Normalized waveforms.
    freqs : Tensor [N, n_components]
        Integer frequencies sampled in the generator support.
    amps : Tensor [N, n_components]
        Per-component amplitudes after any optional global gain stress and before
        waveform normalization.
    phases : Tensor [N, n_components]
        Per-component phases in radians.
    """
    if normalize_divisor <= 0.0:
        raise ValueError("normalize_divisor must be > 0")
    if gain_distribution not in {"none", "uniform", "log_uniform"}:
        raise ValueError(f"Unsupported gain_distribution: {gain_distribution}")
    if gain_min <= 0.0:
        raise ValueError("gain_min must be > 0")
    if gain_max < gain_min:
        raise ValueError("gain_max must be >= gain_min")

    rng = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 2 * torch.pi, length)

    signals = []
    all_freqs = []
    all_amps = []
    all_phases = []
    for _ in range(n_samples):
        freqs = torch.randint(1, 20, (n_components,), generator=rng).float()
        amps = torch.rand(n_components, generator=rng) + 0.3
        phases = torch.rand(n_components, generator=rng) * 2 * torch.pi
        if gain_distribution == "uniform":
            gain = gain_min + (gain_max - gain_min) * torch.rand((), generator=rng)
        elif gain_distribution == "log_uniform":
            log_min = math.log(gain_min)
            log_max = math.log(gain_max)
            gain = torch.exp(log_min + (log_max - log_min) * torch.rand((), generator=rng))
        else:
            gain = torch.tensor(1.0)
        amps = amps * gain
        sig = sum(a * torch.sin(f * t + p) for a, f, p in zip(amps, freqs, phases))
        signals.append(sig)
        all_freqs.append(freqs)
        all_amps.append(amps)
        all_phases.append(phases)

    X = torch.stack(signals).unsqueeze(1)  # [N, 1, T]

    X = X / normalize_divisor

    return (
        X,
        torch.stack(all_freqs),
        torch.stack(all_amps),
        torch.stack(all_phases),
    )


def make_toy_data(
    n_samples: int = 2000,
    length: int = 128,
    n_components: int = 3,
    seed: int = 0,
    gain_distribution: str = "none",
    gain_min: float = 1.0,
    gain_max: float = 1.0,
    normalize_divisor: float = 4.0,
) -> TensorDataset:
    """Sum of *n_components* continuous sinusoids.

    Simple baseline: get this working first (k should drop below 1),
    then graduate to intermittent/gated signals.
    """
    X, _, _, _ = generate_toy_sinusoid_tensors(
        n_samples=n_samples,
        length=length,
        n_components=n_components,
        seed=seed,
        gain_distribution=gain_distribution,
        gain_min=gain_min,
        gain_max=gain_max,
        normalize_divisor=normalize_divisor,
    )
    return TensorDataset(X)


def _mean_active_per_frame(mask: torch.Tensor) -> float:
    return mask.float().sum(dim=1).mean().item()


def _mean_active_total(mask: torch.Tensor) -> float:
    if mask.dim() == 3:
        return mask.any(dim=2).float().sum(dim=1).mean().item()
    return _mean_active_per_frame(mask)


def _squeeze_binary_projection(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4 and tensor.shape[-2] == 1:
        tensor = tensor.squeeze(-2)
    if tensor.dim() == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def _maybe_run_sinusoid_sparse_monitor(
    *,
    cfg: TrainConfig,
    model: HybridSparseVAE,
    epoch: int,
) -> dict[str, float] | None:
    if cfg.dataset != "sinusoid" or cfg.sinusoid_sparse_monitor_every <= 0:
        return None
    should_run = (epoch % cfg.sinusoid_sparse_monitor_every == 0) or (epoch == cfg.epochs)
    if not should_run:
        return None

    repo_root = Path(__file__).resolve().parent
    checkpoint_path = Path(cfg.save_dir) / "hybrid_vae_monitor_latest.pt"
    output_json = Path(cfg.save_dir) / "sparse_recovery_monitor_latest.json"
    hydra_config_path = Path.cwd() / ".hydra" / "config.yaml"

    torch.save(model.state_dict(), checkpoint_path)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_sparse_recovery.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--k-min",
        str(cfg.k_min),
        "--n-samples",
        str(cfg.sinusoid_sparse_monitor_samples),
        "--length",
        str(cfg.input_length),
        "--n-components",
        str(cfg.sinusoid_sparse_monitor_components),
        "--seed",
        str(cfg.sinusoid_sparse_monitor_seed),
        "--max-frequency",
        str(cfg.sinusoid_sparse_monitor_max_frequency),
        "--gain-distribution",
        str(cfg.sinusoid_gain_distribution),
        "--gain-min",
        str(cfg.sinusoid_gain_min),
        "--gain-max",
        str(cfg.sinusoid_gain_max),
        "--normalize-divisor",
        str(cfg.sinusoid_normalize_divisor),
        "--output-json",
        str(output_json),
    ]
    if hydra_config_path.exists():
        cmd.extend(["--hydra-config", str(hydra_config_path)])

    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(
            "Sparse monitor failed at epoch %d with code %s\nstdout:\n%s\nstderr:\n%s",
            epoch,
            exc.returncode,
            exc.stdout,
            exc.stderr,
        )
        return None

    payload: dict[str, object]
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            log.warning("Sparse monitor emitted non-JSON stdout at epoch %d: %s", epoch, completed.stdout)
            return None

    support_eval = payload.get("support_eval", {})
    latents = payload.get("latents", {})
    reconstruction = payload.get("reconstruction", {})
    if not isinstance(support_eval, dict) or not isinstance(latents, dict) or not isinstance(reconstruction, dict):
        log.warning("Sparse monitor returned malformed payload at epoch %d", epoch)
        return None

    return {
        "sparse_monitor_support_f1": float(support_eval.get("support_f1_mean", 0.0)),
        "sparse_monitor_support_precision": float(support_eval.get("support_precision_mean", 0.0)),
        "sparse_monitor_support_recall": float(support_eval.get("support_recall_mean", 0.0)),
        "sparse_monitor_recon_mse": float(reconstruction.get("recon_mse_per_example", 0.0)),
        "sparse_monitor_n_active_frame": float(latents.get("n_active_frame_mean", 0.0)),
        "sparse_monitor_collapse": float(bool(latents.get("collapsed", False))),
    }


def compute_effective_atom_matrix(model: HybridSparseVAE, cfg: TrainConfig) -> torch.Tensor:
    """Return effective atoms in signal space for coherence regularization.

    For strict linear decoders with a learned dictionary, penalize the composed
    basis seen by the reconstruction loss. For ConvNMF, flatten time-frequency
    motifs. Otherwise fall back to the dictionary atoms.
    """
    if cfg.decoder_type in {"linear", "linear_positive"} and not model.temporal_mode:
        dict_w = model.latent.dictionary.weight
        if cfg.normalize_dict:
            dict_w = F.normalize(dict_w, p=2, dim=0)
        decoder_layer = model.decoder.net[0] if isinstance(model.decoder.net, torch.nn.Sequential) else model.decoder.net
        dec_w = decoder_layer.weight
        if cfg.decoder_type == "linear_positive":
            dec_w = F.softplus(dec_w)
        eff = dec_w @ dict_w  # [signal_dim, n_atoms]
        return F.normalize(eff, p=2, dim=0)

    if model.temporal_mode:
        atoms = model.decoder.weight
        atoms = atoms.reshape(atoms.shape[0], -1).T  # [feat_dim, n_atoms]
        return F.normalize(atoms, p=2, dim=0)

    return model.get_dict_atoms().to(next(model.parameters()).device)


def coherence_regularizer(model: HybridSparseVAE, cfg: TrainConfig) -> torch.Tensor:
    """Mean squared off-diagonal Gram penalty on normalized effective atoms."""
    atoms = compute_effective_atom_matrix(model, cfg)
    gram = atoms.T @ atoms
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    off_diag = gram - eye
    return off_diag.pow(2).mean()


# ---------------------------------------------------------------------------
#  KL schedule
# ---------------------------------------------------------------------------

def get_phase(epoch: int, cfg: TrainConfig) -> tuple[str, float]:
    """4-phase schedule with optional late soft fine-tuning.

    Default:
        P1: soft, beta=0
        P2: stochastic, beta=0
        P3: stochastic, beta ramp
        P4: stochastic, beta=max

    Optional:
        P5: soft, beta=max (activated when ``soft_finetune_start > 0``)
    """
    if epoch <= cfg.phase1_end:
        return "soft", 0.0
    if epoch <= cfg.phase2_end:
        return "stochastic", 0.0
    if epoch <= cfg.phase3_end:
        # Phase 3: KL ramp 0 → 1.0
        ramp_len = max(cfg.phase3_end - cfg.phase2_end, 1)
        frac = (epoch - cfg.phase2_end) / ramp_len
        return "stochastic", min(frac, 1.0)
    if cfg.soft_finetune_start > 0 and epoch >= cfg.soft_finetune_start:
        return "soft", 1.0
    # Phase 4: stationary at β=final
    return "stochastic", 1.0


def _pit_two_source_si_sdr(
    mix_wav: torch.Tensor,
    pred1_wav: torch.Tensor,
    pred2_wav: torch.Tensor,
    target1_wav: torch.Tensor,
    target2_wav: torch.Tensor,
) -> tuple[float, float]:
    from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

    mix_ref = 0.5 * (
        scale_invariant_signal_distortion_ratio(mix_wav, target1_wav)
        + scale_invariant_signal_distortion_ratio(mix_wav, target2_wav)
    )
    perm_a = 0.5 * (
        scale_invariant_signal_distortion_ratio(pred1_wav, target1_wav)
        + scale_invariant_signal_distortion_ratio(pred2_wav, target2_wav)
    )
    perm_b = 0.5 * (
        scale_invariant_signal_distortion_ratio(pred1_wav, target2_wav)
        + scale_invariant_signal_distortion_ratio(pred2_wav, target1_wav)
    )
    best_sep = torch.maximum(perm_a, perm_b)
    return float(best_sep.item()), float((best_sep - mix_ref).item())


def _audio_for_wandb(audio: torch.Tensor, norm_denom: torch.Tensor) -> torch.Tensor:
    safe_audio = torch.nan_to_num(audio.detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(norm_denom.detach().cpu().item()) if torch.is_tensor(norm_denom) else float(norm_denom)
    if not math.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    return safe_audio.div(denom).clamp_(-1.0, 1.0)


def _make_bss_spectrogram_panel(
    mix_mag: torch.Tensor,
    s1_ref_mag: torch.Tensor,
    s2_ref_mag: torch.Tensor,
    s1_est_mag: torch.Tensor,
    s2_est_mag: torch.Tensor,
    utt_id: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Mix |X|", mix_mag),
        ("GT S1", s1_ref_mag),
        ("Est S1", s1_est_mag),
        ("Est Sum", s1_est_mag + s2_est_mag),
        ("GT S2", s2_ref_mag),
        ("Est S2", s2_est_mag),
    ]
    log_panels = [torch.log10(p.detach().cpu().float().clamp_min(1e-6)) for _, p in panels]
    vmin = min(float(p.min().item()) for p in log_panels)
    vmax = max(float(p.max().item()) for p in log_panels)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (title, _), panel in zip(axes.flat, panels, log_panels):
        im = ax.imshow(panel.numpy(), aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")
    fig.suptitle(f"LibriMix BSS monitor: {utt_id}")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="log10 magnitude")
    return fig


def _maybe_log_librimix_wandb_monitor(
    cfg: TrainConfig,
    model: HybridSparseVAE,
    epoch: int,
    device: torch.device,
    monitor_loader,
) -> None:
    if not cfg.use_wandb or monitor_loader is None:
        return

    should_eval = cfg.wandb_bss_eval_every > 0 and (
        epoch == 1 or epoch % cfg.wandb_bss_eval_every == 0
    )
    if not should_eval:
        return

    should_log_media = cfg.wandb_bss_media_every > 0 and (
        epoch == 1 or epoch % cfg.wandb_bss_media_every == 0
    )

    import wandb
    from scripts.inference_bss import separate_sources
    from utils.separation import wiener_separation

    model_was_training = model.training
    t0 = time.perf_counter()
    si_sdr_vals: list[float] = []
    si_sdri_vals: list[float] = []
    oracle_si_sdr_vals: list[float] = []
    oracle_si_sdri_vals: list[float] = []
    payload: dict[str, object] = {}

    try:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(monitor_loader):
                if cfg.wandb_bss_max_eval > 0 and i >= cfg.wandb_bss_max_eval:
                    break

                mix_mag = batch["mixture_mag"].to(device)
                mix_c = batch["mixture_complex"].to(device)
                mix_wav = batch["mixture_wav"].to(device)
                s1_wav = batch["source1_wav"].to(device)
                s2_wav = batch["source2_wav"].to(device)
                s1_mag = batch["source1_mag"].to(device)
                s2_mag = batch["source2_mag"].to(device)
                lengths = batch["lengths"].to(device)
                utt_id = batch["utt_id"][0]

                out = separate_sources(
                    model=model,
                    mix_mag=mix_mag,
                    mix_complex=mix_c,
                    alpha=cfg.wandb_bss_alpha,
                    h_representation=cfg.wandb_bss_h_representation,
                    mask_power=cfg.wandb_bss_mask_power,
                    n_fft=cfg.librimix_n_fft,
                    hop_length=cfg.librimix_hop_length,
                    win_length=cfg.librimix_win_length,
                    length=int(lengths[0].item()),
                    cluster_method=cfg.wandb_bss_cluster_method,
                )

                l = int(lengths[0].item())
                si_sdr, si_sdri = _pit_two_source_si_sdr(
                    mix_wav=mix_wav[0, :l],
                    pred1_wav=out["source1_waveform"][0, :l].to(device),
                    pred2_wav=out["source2_waveform"][0, :l].to(device),
                    target1_wav=s1_wav[0, :l],
                    target2_wav=s2_wav[0, :l],
                )
                si_sdr_vals.append(si_sdr)
                si_sdri_vals.append(si_sdri)

                oracle_sep = wiener_separation(
                    mixture_complex=mix_c,
                    source1_mag=s1_mag,
                    source2_mag=s2_mag,
                    n_fft=cfg.librimix_n_fft,
                    hop_length=cfg.librimix_hop_length,
                    win_length=cfg.librimix_win_length,
                    length=lengths,
                    mask_power=cfg.wandb_bss_mask_power,
                )
                oracle_si_sdr, oracle_si_sdri = _pit_two_source_si_sdr(
                    mix_wav=mix_wav[0, :l],
                    pred1_wav=oracle_sep["source1_waveform"][0, :l].to(device),
                    pred2_wav=oracle_sep["source2_waveform"][0, :l].to(device),
                    target1_wav=s1_wav[0, :l],
                    target2_wav=s2_wav[0, :l],
                )
                oracle_si_sdr_vals.append(oracle_si_sdr)
                oracle_si_sdri_vals.append(oracle_si_sdri)

                if should_log_media and i == 0:
                    mix_peak = mix_wav[0, :l].abs().max().clamp_min(1e-8)
                    mix_est = out["source1_waveform"][0, :l] + out["source2_waveform"][0, :l]
                    fig = _make_bss_spectrogram_panel(
                        mix_mag=mix_mag[0],
                        s1_ref_mag=s1_mag[0],
                        s2_ref_mag=s2_mag[0],
                        s1_est_mag=out["source1_mag"][0],
                        s2_est_mag=out["source2_mag"][0],
                        utt_id=utt_id,
                    )
                    payload["monitor_bss/spectrogram"] = wandb.Image(fig)
                    payload["monitor_bss/audio_mix"] = wandb.Audio(
                        _audio_for_wandb(mix_wav[0, :l], mix_peak).numpy(),
                        sample_rate=cfg.librimix_sample_rate,
                        caption=f"{utt_id} mix",
                    )
                    payload["monitor_bss/audio_mix_est"] = wandb.Audio(
                        _audio_for_wandb(mix_est, mix_peak).numpy(),
                        sample_rate=cfg.librimix_sample_rate,
                        caption=f"{utt_id} estimated mixture",
                    )
                    payload["monitor_bss/audio_s1_est"] = wandb.Audio(
                        _audio_for_wandb(out["source1_waveform"][0, :l], mix_peak).numpy(),
                        sample_rate=cfg.librimix_sample_rate,
                        caption=f"{utt_id} estimated source 1",
                    )
                    payload["monitor_bss/audio_s2_est"] = wandb.Audio(
                        _audio_for_wandb(out["source2_waveform"][0, :l], mix_peak).numpy(),
                        sample_rate=cfg.librimix_sample_rate,
                        caption=f"{utt_id} estimated source 2",
                    )
                    payload["monitor_bss/audio_s1_ref"] = wandb.Audio(
                        _audio_for_wandb(s1_wav[0, :l], mix_peak).numpy(),
                        sample_rate=cfg.librimix_sample_rate,
                        caption=f"{utt_id} reference source 1",
                    )
                    payload["monitor_bss/audio_s2_ref"] = wandb.Audio(
                        _audio_for_wandb(s2_wav[0, :l], mix_peak).numpy(),
                        sample_rate=cfg.librimix_sample_rate,
                        caption=f"{utt_id} reference source 2",
                    )
                    payload["monitor_bss/sample_si_sdr_db"] = si_sdr
                    payload["monitor_bss/sample_si_sdri_db"] = si_sdri
                    payload["monitor_bss/sample_oracle_si_sdr_db"] = oracle_si_sdr
                    payload["monitor_bss/sample_oracle_si_sdri_db"] = oracle_si_sdri
                    payload["monitor_bss/sample_oracle_gap_db"] = oracle_si_sdri - si_sdri
                    payload["monitor_bss/sample_cluster_size_1"] = len(out["omega_1"])
                    payload["monitor_bss/sample_cluster_size_2"] = len(out["omega_2"])
                    payload["monitor_bss/sample_utt_id"] = utt_id
                    payload["monitor_bss/audio_s1_oracle"] = wandb.Audio(
                        _audio_for_wandb(oracle_sep["source1_waveform"][0, :l], mix_peak).numpy(),
                        sample_rate=cfg.librimix_sample_rate,
                        caption=f"{utt_id} oracle source 1",
                    )
                    payload["monitor_bss/audio_s2_oracle"] = wandb.Audio(
                        _audio_for_wandb(oracle_sep["source2_waveform"][0, :l], mix_peak).numpy(),
                        sample_rate=cfg.librimix_sample_rate,
                        caption=f"{utt_id} oracle source 2",
                    )
                    import matplotlib.pyplot as plt

                    plt.close(fig)

        if si_sdr_vals:
            payload["monitor_bss/si_sdr_mean_db"] = float(sum(si_sdr_vals) / len(si_sdr_vals))
            payload["monitor_bss/si_sdri_mean_db"] = float(sum(si_sdri_vals) / len(si_sdri_vals))
            payload["monitor_bss/num_eval_examples"] = int(len(si_sdri_vals))
        if oracle_si_sdr_vals:
            oracle_si_sdr_mean = float(sum(oracle_si_sdr_vals) / len(oracle_si_sdr_vals))
            oracle_si_sdri_mean = float(sum(oracle_si_sdri_vals) / len(oracle_si_sdri_vals))
            payload["monitor_bss/oracle_si_sdr_mean_db"] = oracle_si_sdr_mean
            payload["monitor_bss/oracle_si_sdri_mean_db"] = oracle_si_sdri_mean
            if si_sdri_vals:
                payload["monitor_bss/oracle_gap_db"] = oracle_si_sdri_mean - float(sum(si_sdri_vals) / len(si_sdri_vals))
        payload["monitor_bss/runtime_sec"] = time.perf_counter() - t0
        wandb.log(payload, step=epoch)
    except Exception as exc:
        log.warning("W&B LibriMix monitor failed at epoch %d: %s", epoch, exc)
    finally:
        if model_was_training:
            model.train()


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
    if (cfg.encoder_type == "fully_polar_lista" or cfg.fully_polar_encoder) and cfg.beta_delta_final != 0.0:
        raise ValueError(
            "fully_polar_lista uses beta_presence_final/beta_sign_final; set beta_delta_final=0.0 to avoid ambiguity."
        )

    # ---- Data ----------------------------------------------------------
    dataset_name = cfg.dataset.lower()
    
    dataset = None
    loader = None
    monitor_loader = None

    if dataset_name == "sinusoid":
        dataset = make_toy_data(
            n_samples=2000,
            length=cfg.input_length,
            n_components=3,
            seed=cfg.seed,
            gain_distribution=cfg.sinusoid_gain_distribution,
            gain_min=cfg.sinusoid_gain_min,
            gain_max=cfg.sinusoid_gain_max,
            normalize_divisor=cfg.sinusoid_normalize_divisor,
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
            use_instance_norm=cfg.spectrogram_enhancements,
            denoise=cfg.denoise,
        )
    elif dataset_name == "librimix":
        from data.datasets import get_librimix_dataset, get_librimix_dataloader

        librimix_root = (
            cfg.librimix_root_dir if cfg.librimix_root_dir else f"{cfg.data_dir}/Libri2Mix"
        )
        dataset = get_librimix_dataset(
            root_dir=librimix_root,
            split=cfg.librimix_split,
            sample_rate=cfg.librimix_sample_rate,
            mix_type=cfg.librimix_mix_type,
            mixture_dirname=cfg.librimix_mixture_dirname,
            n_fft=cfg.librimix_n_fft,
            hop_length=cfg.librimix_hop_length,
            win_length=cfg.librimix_win_length,
            max_frames=cfg.max_frames,
            crop_mode=cfg.librimix_crop_mode,
        )
        loader = get_librimix_dataloader(
            root_dir=librimix_root,
            split=cfg.librimix_split,
            sample_rate=cfg.librimix_sample_rate,
            mix_type=cfg.librimix_mix_type,
            mixture_dirname=cfg.librimix_mixture_dirname,
            n_fft=cfg.librimix_n_fft,
            hop_length=cfg.librimix_hop_length,
            win_length=cfg.librimix_win_length,
            max_frames=cfg.max_frames,
            crop_mode=cfg.librimix_crop_mode,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        if cfg.use_wandb and (
            cfg.wandb_bss_eval_every > 0 or cfg.wandb_bss_media_every > 0
        ):
            try:
                monitor_loader = get_librimix_dataloader(
                    root_dir=librimix_root,
                    split=cfg.wandb_bss_split,
                    sample_rate=cfg.librimix_sample_rate,
                    mix_type=cfg.librimix_mix_type,
                    mixture_dirname=cfg.librimix_mixture_dirname,
                    n_fft=cfg.librimix_n_fft,
                    hop_length=cfg.librimix_hop_length,
                    win_length=cfg.librimix_win_length,
                    max_frames=None,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
            except Exception as exc:
                log.warning("Could not initialize LibriMix W&B monitor loader: %s", exc)
                monitor_loader = None
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    if loader is None:
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # ---- Auto-detect shape to avoid magic constants --------------------
    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, dict):
            sample_x = sample["mixture_mag"]
        elif isinstance(sample, (tuple, list)):
            sample_x = sample[0]
        else:
            sample_x = sample
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
        k_max=cfg.k_max,
        magnitude_dist=cfg.magnitude_dist,
        structure_mode=cfg.structure_mode,
        motif_width=cfg.motif_width,
        decoder_stride=cfg.decoder_stride if cfg.decoder_stride > 0 else cfg.input_length // 4,
        match_encoder_decoder_stride=cfg.match_encoder_decoder_stride,
        lista_iterations=cfg.lista_iterations,
        lista_threshold_init=cfg.lista_threshold_init,
        delta_head_mode=cfg.delta_head_mode,
        polar_encoder=cfg.polar_encoder,
        fully_polar_encoder=cfg.fully_polar_encoder,
        shape_norm=cfg.shape_norm,
        gain_feature=cfg.gain_feature,
        gamma_scale_injection=cfg.gamma_scale_injection,
        shape_detach_to_gamma=cfg.shape_detach_to_gamma,
        delta_factorization=cfg.delta_factorization,
        presence_estimator=cfg.presence_estimator,
        sign_estimator=cfg.sign_estimator,
        presence_alpha=cfg.presence_alpha,
        tau_presence_eval=cfg.tau_presence_eval,
        sign_tau_eval=cfg.sign_tau_eval,
        presence_head_bias_init=cfg.presence_head_bias_init,
        sign_head_bias_init=cfg.sign_head_bias_init,
        gumbel_epsilon=cfg.gumbel_epsilon,
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
    fully_polar_mode = (cfg.encoder_type == "fully_polar_lista") or bool(cfg.fully_polar_encoder)

    # Store initial latent and effective dictionaries for drift measurement.
    dict_init_snapshot = model.get_dict_atoms().clone()
    eff_init_snapshot = compute_effective_atom_matrix(model, cfg).detach().clone()

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
        eff_beta_presence = beta_frac * cfg.beta_presence_final
        eff_beta_sign = beta_frac * cfg.beta_sign_final
        eff_lambda_presence_consistency = beta_frac * cfg.lambda_presence_consistency_final

        # Progressive lambda_silence: ramps from lambda_silence → lambda_silence_final
        # over the same schedule as beta (Phase 3 ramp + Phase 4 plateau).
        # P1/P2: beta_frac=0 → use initial value; P4: beta_frac=1 → use final value.
        eff_lambda_silence = (
            cfg.lambda_silence + beta_frac * (cfg.lambda_silence_final - cfg.lambda_silence)
        )

        # Phase label for logging
        if epoch <= cfg.phase1_end:
            phase_label = "P1:soft"
        elif epoch <= cfg.phase2_end:
            phase_label = "P2:stoch"
        elif epoch <= cfg.phase3_end:
            phase_label = "P3:ramp"
        elif cfg.soft_finetune_start > 0 and epoch >= cfg.soft_finetune_start:
            phase_label = "P5:softft"
        else:
            phase_label = "P4:conv"

        epoch_loss = 0.0
        epoch_metrics: dict[str, float] = {
            "recon": 0.0,
            "kl_gamma": 0.0,
            "kl_delta": 0.0,
            "kl_presence": 0.0,
            "kl_sign": 0.0,
            "kl_gamma_raw": 0.0,
            "kl_delta_raw": 0.0,
            "kl_presence_raw": 0.0,
            "kl_sign_raw": 0.0,
            "weighted_kl_gamma": 0.0,
            "weighted_kl_delta": 0.0,
            "weighted_kl_presence": 0.0,
            "weighted_kl_sign": 0.0,
            "presence_consistency": 0.0,
            "presence_consistency_raw": 0.0,
            "weighted_presence_consistency": 0.0,
            "kl_norm_factor": 0.0,
            "coherence": 0.0,
        }
        epoch_monitor_metrics: dict[str, float] = {
            "k_mean": 0.0,
            "k_active": 0.0,
            "n_active_frame": 0.0,
            "n_active_total": 0.0,
            "n_active_frame_det": 0.0,
            "n_active_total_det": 0.0,
            "expected_nonzero_frame": 0.0,
            "sparsity": 0.0,
            "presence_sample_frame": 0.0,
            "presence_sample_total": 0.0,
            "presence_argmax_frame": 0.0,
            "presence_argmax_total": 0.0,
            "presence_prob_frame": 0.0,
            "presence_sample_argmax_agreement": 0.0,
            "presence_logit_margin_mean": 0.0,
        }

        for batch in loader:
            if isinstance(batch, dict):
                batch_x = batch["mixture_mag"]
            else:
                batch_x = batch[0]
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_recon, info = model(batch_x, temp=temp, sampling=sampling_mode)

            if fully_polar_mode:
                presence_prior_t = torch.tensor(
                    [float(v) for v in cfg.presence_prior.split(",")],
                    device=device,
                    dtype=batch_x.dtype,
                )
                sign_prior_t = torch.tensor(
                    [float(v) for v in cfg.sign_prior.split(",")],
                    device=device,
                    dtype=batch_x.dtype,
                )
                loss, metrics = compute_fully_polar_loss(
                    x=batch_x,
                    x_recon=x_recon,
                    params=info,
                    k_0=cfg.k_0,
                    theta_0=cfg.theta_0,
                    presence_prior_probs=presence_prior_t,
                    sign_prior_probs=sign_prior_t,
                    beta_gamma=eff_beta_gamma,
                    beta_presence=eff_beta_presence,
                    beta_sign=eff_beta_sign,
                    masked_recon=cfg.masked_recon,
                    lambda_silence=eff_lambda_silence,
                    lambda_recon_l1=cfg.lambda_recon_l1,
                    kl_normalization=cfg.kl_normalization,
                    gamma_kl_target=cfg.gamma_kl_target,
                    lambda_presence_consistency=eff_lambda_presence_consistency,
                    presence_consistency_target=cfg.presence_consistency_target,
                )
            else:
                # Parse delta prior
                dp = [float(v) for v in cfg.delta_prior.split(",")]
                
                # Auto-adapt prior to structure mode
                logits_classes = info["logits"].shape[-1]
                if logits_classes == 2 and len(dp) == 3:
                    dp = [dp[1], dp[0] + dp[2]]  # [P(0), P(-1) + P(+1)]
                elif logits_classes == 3 and len(dp) == 2:
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
                    masked_recon=cfg.masked_recon,
                    lambda_silence=eff_lambda_silence,
                    lambda_recon_l1=cfg.lambda_recon_l1,
                    kl_normalization=cfg.kl_normalization,
                )

            coherence_pen = torch.tensor(0.0, device=device)
            if cfg.lambda_atom_coherence > 0.0:
                coherence_pen = coherence_regularizer(model, cfg)
                loss = loss + (beta_frac * cfg.lambda_atom_coherence) * coherence_pen

            loss.backward()
            
            # Separate gradient clipping for dictionary stability
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=5.0)
            if not dict_frozen:
                torch.nn.utils.clip_grad_norm_(dict_params, max_norm=cfg.gradient_clip_dict)
                
            optimizer.step()

            epoch_loss += loss.item()
            for key in (
                "recon",
                "kl_gamma",
                "kl_delta",
                "kl_presence",
                "kl_sign",
                "kl_gamma_raw",
                "kl_delta_raw",
                "kl_presence_raw",
                "kl_sign_raw",
                "weighted_kl_gamma",
                "weighted_kl_delta",
                "weighted_kl_presence",
                "weighted_kl_sign",
                "presence_consistency",
                "presence_consistency_raw",
                "weighted_presence_consistency",
                "kl_norm_factor",
            ):
                epoch_metrics[key] += metrics.get(key, torch.tensor(0.0, device=device)).item()
            epoch_metrics["coherence"] += coherence_pen.detach().item()

            k_all = info["k"].detach()
            delta = info["delta"].detach()
            active_mask = (delta != 0)
            epoch_monitor_metrics["k_mean"] += k_all.mean().item()
            epoch_monitor_metrics["k_active"] += (
                k_all[active_mask].mean().item() if active_mask.any() else 0.0
            )
            n_active_per_frame = _mean_active_per_frame(active_mask)
            n_active_total = _mean_active_total(active_mask)
            if "presence_probs" in info:
                presence_probs = _squeeze_binary_projection(info["presence_probs"].detach())
                sampled_presence_mask = _squeeze_binary_projection(
                    info.get("presence_det", (presence_probs > cfg.tau_presence_eval)).detach()
                ).bool()
                presence_logits = _squeeze_binary_projection(
                    info.get("presence_logits", info["logits"]).detach()
                )
                det_active_mask = (presence_logits.argmax(dim=-1) == 1)
                expected_nonzero = presence_probs
                epoch_monitor_metrics["presence_sample_frame"] += _mean_active_per_frame(sampled_presence_mask)
                epoch_monitor_metrics["presence_sample_total"] += _mean_active_total(sampled_presence_mask)
                epoch_monitor_metrics["presence_argmax_frame"] += _mean_active_per_frame(det_active_mask)
                epoch_monitor_metrics["presence_argmax_total"] += _mean_active_total(det_active_mask)
                epoch_monitor_metrics["presence_prob_frame"] += expected_nonzero.sum(dim=1).mean().item()
                epoch_monitor_metrics["presence_sample_argmax_agreement"] += (
                    (sampled_presence_mask == det_active_mask).float().mean().item()
                )
                margin = presence_logits[..., 1] - presence_logits[..., 0]
                epoch_monitor_metrics["presence_logit_margin_mean"] += margin.mean().item()
            else:
                logits = info["logits"].detach()
                if logits.dim() == 4 and logits.shape[-2] == 1:
                    logits = logits.squeeze(-2)
                probs = F.softmax(logits / max(temp, 1e-6), dim=-1)
                zero_index = 1 if logits.shape[-1] == 3 else 0
                det_active_mask = logits.argmax(dim=-1) != zero_index
                zero_prob = probs[..., zero_index]
                expected_nonzero = 1.0 - zero_prob
            n_active_per_frame_det = _mean_active_per_frame(det_active_mask)
            n_active_total_det = _mean_active_total(det_active_mask)
            expected_nonzero_frame = expected_nonzero.sum(dim=1).mean().item()
            epoch_monitor_metrics["n_active_frame"] += n_active_per_frame
            epoch_monitor_metrics["n_active_total"] += n_active_total
            epoch_monitor_metrics["n_active_frame_det"] += n_active_per_frame_det
            epoch_monitor_metrics["n_active_total_det"] += n_active_total_det
            epoch_monitor_metrics["expected_nonzero_frame"] += expected_nonzero_frame
            epoch_monitor_metrics["sparsity"] += (delta == 0).float().mean().item()

        n_batches = len(loader)
        avg = {k: v / n_batches for k, v in epoch_metrics.items()}
        mon = {k: v / n_batches for k, v in epoch_monitor_metrics.items()}
        k_mean = mon["k_mean"]
        k_active = mon["k_active"]
        n_active_per_frame = mon["n_active_frame"]
        n_active_total = mon["n_active_total"]
        n_active_per_frame_det = mon["n_active_frame_det"]
        n_active_total_det = mon["n_active_total_det"]
        expected_nonzero_frame = mon["expected_nonzero_frame"]
        sparsity = mon["sparsity"]
        presence_sample_frame = mon["presence_sample_frame"]
        presence_sample_total = mon["presence_sample_total"]
        presence_argmax_frame = mon["presence_argmax_frame"]
        presence_argmax_total = mon["presence_argmax_total"]
        presence_prob_frame = mon["presence_prob_frame"]
        presence_sample_argmax_agreement = mon["presence_sample_argmax_agreement"]
        presence_logit_margin_mean = mon["presence_logit_margin_mean"]
        collapse_flag = float((n_active_per_frame < 1.0) and (n_active_total < 1.0))
        current_atoms = model.get_dict_atoms()
        dict_drift = 1.0 - F.cosine_similarity(
            current_atoms.flatten().unsqueeze(0),
            dict_init_snapshot.flatten().unsqueeze(0),
        ).item()
        current_effective_atoms = compute_effective_atom_matrix(model, cfg)
        effective_atom_drift = 1.0 - F.cosine_similarity(
            current_effective_atoms.flatten().unsqueeze(0),
            eff_init_snapshot.flatten().unsqueeze(0),
        ).item()
        sparse_monitor_metrics = _maybe_run_sinusoid_sparse_monitor(
            cfg=cfg,
            model=model,
            epoch=epoch,
        )

        if epoch % cfg.log_every == 0 or epoch == 1 or sparse_monitor_metrics is not None:
            extra_presence_log = ""
            if presence_prob_frame > 0.0 or presence_argmax_frame > 0.0 or presence_sample_frame > 0.0:
                extra_presence_log = (
                    f"  pres_sample={presence_sample_frame:.1f}/{presence_sample_total:.1f}  "
                    f"pres_argmax={presence_argmax_frame:.1f}/{presence_argmax_total:.1f}  "
                    f"pres_mass={presence_prob_frame:.1f}  "
                    f"pres_agree={presence_sample_argmax_agreement:.2%}  "
                    f"pres_margin={presence_logit_margin_mean:.3f}"
                )
            sparse_log = ""
            if sparse_monitor_metrics is not None:
                sparse_log = (
                    f"  sr_f1={sparse_monitor_metrics['sparse_monitor_support_f1']:.3f}  "
                    f"sr_p={sparse_monitor_metrics['sparse_monitor_support_precision']:.3f}  "
                    f"sr_r={sparse_monitor_metrics['sparse_monitor_support_recall']:.3f}  "
                    f"sr_act={sparse_monitor_metrics['sparse_monitor_n_active_frame']:.1f}  "
                    f"sr_collapse={int(sparse_monitor_metrics['sparse_monitor_collapse'])}"
                )
            log.info(
                f"Epoch {epoch:4d} [{phase_label}] | "
                f"loss {epoch_loss/n_batches:.4f} | "
                f"recon {avg['recon']:.4f} | kl_γ {avg['kl_gamma']:.4f} | "
                f"kl_δ {avg['kl_delta']:.4f} | wkl_γ {avg['weighted_kl_gamma']:.4f} | "
                f"wkl_δ {avg['weighted_kl_delta']:.4f} | "
                f"pcons {avg['presence_consistency']:.4f} | "
                f"wpcons {avg['weighted_presence_consistency']:.4f} | "
                f"coh {avg['coherence']:.4f} | "
                f"k̄={k_mean:.3f}  k_act={k_active:.3f}  "
                f"n_act_frame={n_active_per_frame:.1f}  n_act_total={n_active_total:.1f}/{cfg.n_atoms}  "
                f"n_act_det={n_active_per_frame_det:.1f}  n_act_det_total={n_active_total_det:.1f}/{cfg.n_atoms}  "
                f"nz_mass={expected_nonzero_frame:.1f}  "
                f"δ₀={sparsity:.2%}  "
                f"collapse={int(collapse_flag)}  β={eff_beta_gamma:.4f}  τ={temp:.3f}  "
                f"Δdict={dict_drift:.4f}  "
                f"Δeff={effective_atom_drift:.4f}"
                f"{extra_presence_log}"
                f"{sparse_log}"
            )

        if cfg.use_wandb:
            import wandb

            payload = {
                "epoch": epoch,
                "loss": epoch_loss / n_batches,
                "recon": avg["recon"],
                "kl_gamma": avg["kl_gamma"],
                "kl_delta": avg["kl_delta"],
                "kl_presence": avg["kl_presence"],
                "kl_sign": avg["kl_sign"],
                "kl_gamma_raw": avg["kl_gamma_raw"],
                "kl_delta_raw": avg["kl_delta_raw"],
                "kl_presence_raw": avg["kl_presence_raw"],
                "kl_sign_raw": avg["kl_sign_raw"],
                "weighted_kl_gamma": avg["weighted_kl_gamma"],
                "weighted_kl_delta": avg["weighted_kl_delta"],
                "weighted_kl_presence": avg["weighted_kl_presence"],
                "weighted_kl_sign": avg["weighted_kl_sign"],
                "presence_consistency": avg["presence_consistency"],
                "presence_consistency_raw": avg["presence_consistency_raw"],
                "weighted_presence_consistency": avg["weighted_presence_consistency"],
                "kl_norm_factor": avg["kl_norm_factor"],
                "coherence": avg["coherence"],
                "k_mean": k_mean,
                "k_active": k_active,
                "n_active_frame": n_active_per_frame,
                "n_active": n_active_total,
                "n_active_frame_det": n_active_per_frame_det,
                "n_active_total_det": n_active_total_det,
                "expected_nonzero_frame": expected_nonzero_frame,
                "sparsity": sparsity,
                "collapse_flag": collapse_flag,
                "presence_sample_frame": presence_sample_frame,
                "presence_sample_total": presence_sample_total,
                "presence_argmax_frame": presence_argmax_frame,
                "presence_argmax_total": presence_argmax_total,
                "presence_prob_frame": presence_prob_frame,
                "presence_sample_argmax_agreement": presence_sample_argmax_agreement,
                "presence_logit_margin_mean": presence_logit_margin_mean,
                "dict_drift": dict_drift,
                "effective_atom_drift": effective_atom_drift,
                "phase_beta_gamma": eff_beta_gamma,
                "phase_beta_delta": eff_beta_delta,
                "phase_beta_presence": eff_beta_presence,
                "phase_beta_sign": eff_beta_sign,
                "phase_lambda_presence_consistency": eff_lambda_presence_consistency,
                "temp": temp,
            }
            if sparse_monitor_metrics is not None:
                payload.update(sparse_monitor_metrics)
            wandb.log(payload, step=epoch)
            _maybe_log_librimix_wandb_monitor(
                cfg=cfg,
                model=model,
                epoch=epoch,
                device=device,
                monitor_loader=monitor_loader,
            )

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
