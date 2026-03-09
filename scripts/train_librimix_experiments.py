"""Train follow-up LibriMix experiments on the existing data pipeline.

Experiments:
  - ``direct_mask``: supervised TF-mask baseline for pipeline calibration
  - ``hybrid_partition``: current encoder/latent/shared-decoder family with
    trainable two-source partitioning and source-level supervision
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import get_librimix_dataloader, get_librimix_dataset
from models import HybridLatentPartitionSeparator, SupervisedTFMaskSeparator
from utils.objectives import kl_categorical, kl_gamma
from utils.separation import wiener_separation

log = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_input_shape(dataset) -> tuple[int, int]:
    sample = dataset[0]["mixture_mag"]
    return int(sample.shape[0]), int(sample.shape[1])


def parse_delta_prior(raw: str, structure_mode: str, device: torch.device) -> torch.Tensor:
    probs = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if structure_mode == "binary":
        if len(probs) == 3:
            probs = [probs[1], probs[0] + probs[2]]
        if len(probs) != 2:
            raise ValueError(f"Binary structure_mode expects 2 prior probabilities, got {probs}")
    else:
        if len(probs) == 2:
            probs = [probs[1] / 2.0, probs[0], probs[1] / 2.0]
        if len(probs) != 3:
            raise ValueError(f"Ternary structure_mode expects 3 prior probabilities, got {probs}")
    prior = torch.tensor(probs, dtype=torch.float32, device=device)
    return prior / prior.sum()


def get_phase(epoch: int, args: argparse.Namespace) -> tuple[str, float]:
    if epoch <= args.phase1_end:
        return "soft", 0.0
    if epoch <= args.phase2_end:
        return "stochastic", 0.0
    if epoch <= args.phase3_end:
        ramp_len = max(args.phase3_end - args.phase2_end, 1)
        frac = (epoch - args.phase2_end) / ramp_len
        return "stochastic", min(frac, 1.0)
    return "stochastic", 1.0


def get_temp(epoch: int, args: argparse.Namespace) -> float:
    if epoch >= args.temp_anneal_epochs:
        return args.temp_min
    frac = epoch / max(args.temp_anneal_epochs, 1)
    return args.temp_init + (args.temp_min - args.temp_init) * frac


def compute_kl_normalizer(k_tensor: torch.Tensor, mode: str) -> float:
    """Return the normalization factor applied to raw latent KL sums."""
    if mode == "batch":
        return float(max(k_tensor.shape[0], 1))
    if mode == "site":
        return float(max(k_tensor.numel(), 1))
    raise ValueError(f"Unknown kl_normalization={mode!r}")


def loss_map(est: torch.Tensor, ref: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return (est - ref).abs().mean(dim=(2, 3))
    if loss_type == "mse":
        return (est - ref).pow(2).mean(dim=(2, 3))
    raise ValueError(f"Unknown loss_type={loss_type}")


def pit_two_source_mag_loss(
    est_sources: torch.Tensor,
    ref_sources: torch.Tensor,
    loss_type: str = "l1",
) -> tuple[torch.Tensor, torch.Tensor]:
    perm_a = loss_map(est_sources[:, 0:1], ref_sources[:, 0:1], loss_type).squeeze(1) + loss_map(
        est_sources[:, 1:2], ref_sources[:, 1:2], loss_type
    ).squeeze(1)
    perm_b = loss_map(est_sources[:, 0:1], ref_sources[:, 1:2], loss_type).squeeze(1) + loss_map(
        est_sources[:, 1:2], ref_sources[:, 0:1], loss_type
    ).squeeze(1)
    use_a = perm_a <= perm_b
    best = torch.where(use_a, perm_a, perm_b)
    perm_index = (~use_a).long()
    return best.mean(), perm_index


def mixture_loss(est_mix: torch.Tensor, mix_mag: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return (est_mix - mix_mag).abs().mean()
    if loss_type == "mse":
        return (est_mix - mix_mag).pow(2).mean()
    raise ValueError(f"Unknown loss_type={loss_type}")


def pit_two_source_si_sdr(
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


def build_model(args: argparse.Namespace, input_channels: int, input_length: int) -> torch.nn.Module:
    if args.experiment == "direct_mask":
        return SupervisedTFMaskSeparator(
            n_freq_bins=input_channels,
            hidden_channels=args.direct_hidden_channels,
            num_blocks=args.direct_num_blocks,
            kernel_size=args.direct_kernel_size,
            dropout=args.direct_dropout,
        )

    if args.experiment == "hybrid_partition":
        return HybridLatentPartitionSeparator(
            input_channels=input_channels,
            input_length=input_length,
            encoder_output_dim=args.encoder_output_dim,
            n_atoms=args.n_atoms,
            latent_dim=args.latent_dim,
            motif_width=args.motif_width,
            decoder_stride=args.decoder_stride,
            encoder_type=args.encoder_type,
            dict_init=args.dict_init,
            normalize_dict=not args.no_normalize_dict,
            k_min=args.k_min,
            k_max=args.k_max,
            magnitude_dist=args.magnitude_dist,
            structure_mode=args.structure_mode,
            match_encoder_decoder_stride=args.match_encoder_decoder_stride,
            assignment_hidden=args.assignment_hidden,
        )

    raise ValueError(f"Unknown experiment={args.experiment}")


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.experiment != "hybrid_partition":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    decoder_params = list(model.backbone.decoder.parameters())
    decoder_ids = {id(p) for p in decoder_params}
    other_params = [p for p in model.parameters() if id(p) not in decoder_ids]
    return torch.optim.Adam(
        [
            {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {
                "params": decoder_params,
                "lr": args.lr * args.dict_lr_mult,
                "weight_decay": args.weight_decay,
                "name": "decoder",
            },
        ]
    )


def run_model(
    model: torch.nn.Module,
    mix_mag: torch.Tensor,
    args: argparse.Namespace,
    *,
    epoch: int | None = None,
    eval_mode: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    aux: dict[str, float] = {}
    if args.experiment == "direct_mask":
        return model(mix_mag), aux

    if eval_mode:
        temp = args.temp_min
        sampling = "deterministic"
        beta_frac = 1.0
    else:
        if epoch is None:
            raise ValueError("epoch is required for hybrid_partition training")
        sampling, beta_frac = get_phase(epoch, args)
        temp = get_temp(epoch, args)

    out = model(mix_mag, temp=temp, sampling=sampling)
    aux["temp"] = temp
    aux["beta_frac"] = beta_frac
    return out, aux


def compute_losses(
    model_out: dict[str, torch.Tensor],
    mix_mag: torch.Tensor,
    source1_mag: torch.Tensor,
    source2_mag: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    beta_frac: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ref_sources = torch.stack([source1_mag, source2_mag], dim=1)
    source_loss, _ = pit_two_source_mag_loss(
        model_out["source_mags"],
        ref_sources,
        loss_type=args.source_loss_type,
    )

    mix_recon = model_out.get("mixture_recon", model_out["source_mags"].sum(dim=1))
    mix_loss = mixture_loss(mix_recon, mix_mag, loss_type=args.mix_loss_type)

    metrics = {
        "source_loss": float(source_loss.detach().item()),
        "mix_loss": float(mix_loss.detach().item()),
    }

    if args.experiment != "hybrid_partition":
        total = args.lambda_source * source_loss + args.lambda_mix * mix_loss
        metrics["total_loss"] = float(total.detach().item())
        return total, metrics

    prior_probs = parse_delta_prior(args.delta_prior, args.structure_mode, device)
    kl_norm = compute_kl_normalizer(model_out["k"], args.kl_normalization)
    kl_g_raw = kl_gamma(model_out["k"], model_out["theta"], args.k0, args.theta0)
    kl_d_raw = kl_categorical(model_out["logits"], prior_probs)
    kl_g = kl_g_raw / kl_norm
    kl_d = kl_d_raw / kl_norm
    eff_beta_gamma = beta_frac * args.beta_gamma_final
    eff_beta_delta = beta_frac * args.beta_delta_final
    weighted_kl_gamma = eff_beta_gamma * kl_g
    weighted_kl_delta = eff_beta_delta * kl_d
    total = (
        args.lambda_source * source_loss
        + args.lambda_mix * mix_loss
        + weighted_kl_gamma
        + weighted_kl_delta
    )

    source_assign = model_out["source_assign"].detach()
    assign_entropy = -(source_assign.clamp_min(1e-8) * source_assign.clamp_min(1e-8).log()).sum(dim=1).mean()
    active_mask = model_out["source_acts"].sum(dim=1) > 0
    n_active_frame = active_mask.float().sum(dim=1).mean()
    n_active_total = active_mask.any(dim=2).float().sum(dim=1).mean()

    metrics.update(
        {
            "kl_norm_factor": kl_norm,
            "kl_gamma": float(kl_g.detach().item()),
            "kl_delta": float(kl_d.detach().item()),
            "kl_gamma_raw": float(kl_g_raw.detach().item()),
            "kl_delta_raw": float(kl_d_raw.detach().item()),
            "beta_gamma": eff_beta_gamma,
            "beta_delta": eff_beta_delta,
            "weighted_kl_gamma": float(weighted_kl_gamma.detach().item()),
            "weighted_kl_delta": float(weighted_kl_delta.detach().item()),
            "assign_entropy": float(assign_entropy.item()),
            "n_active_frame": float(n_active_frame.item()),
            "n_active_total": float(n_active_total.item()),
            "total_loss": float(total.detach().item()),
        }
    )
    return total, metrics


def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    si_sdr_vals: list[float] = []
    si_sdri_vals: list[float] = []
    oracle_si_sdri_vals: list[float] = []
    source_loss_vals: list[float] = []
    mix_loss_vals: list[float] = []

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_dev_eval > 0 and i >= args.max_dev_eval:
                break

            mix_mag = batch["mixture_mag"].to(device)
            mix_c = batch["mixture_complex"].to(device)
            mix_wav = batch["mixture_wav"].to(device)
            s1_mag = batch["source1_mag"].to(device)
            s2_mag = batch["source2_mag"].to(device)
            s1_wav = batch["source1_wav"].to(device)
            s2_wav = batch["source2_wav"].to(device)
            lengths = batch["lengths"].to(device)

            out, _ = run_model(model, mix_mag, args, eval_mode=True)
            loss_metrics_total = compute_losses(
                out,
                mix_mag=mix_mag,
                source1_mag=s1_mag,
                source2_mag=s2_mag,
                args=args,
                device=device,
                beta_frac=1.0,
            )[1]
            source_loss_vals.append(loss_metrics_total["source_loss"])
            mix_loss_vals.append(loss_metrics_total["mix_loss"])

            est_sources = out["source_mags"]
            for b in range(mix_mag.shape[0]):
                length_b = int(lengths[b].item())
                sep = wiener_separation(
                    mixture_complex=mix_c[b : b + 1],
                    source1_mag=est_sources[b : b + 1, 0],
                    source2_mag=est_sources[b : b + 1, 1],
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    length=length_b,
                    mask_power=args.mask_power,
                )
                si_sdr, si_sdri = pit_two_source_si_sdr(
                    mix_wav=mix_wav[b, :length_b],
                    pred1_wav=sep["source1_waveform"][0, :length_b],
                    pred2_wav=sep["source2_waveform"][0, :length_b],
                    target1_wav=s1_wav[b, :length_b],
                    target2_wav=s2_wav[b, :length_b],
                )
                si_sdr_vals.append(si_sdr)
                si_sdri_vals.append(si_sdri)

                if args.compute_oracle_gap:
                    oracle = wiener_separation(
                        mixture_complex=mix_c[b : b + 1],
                        source1_mag=s1_mag[b : b + 1],
                        source2_mag=s2_mag[b : b + 1],
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        win_length=args.win_length,
                        length=length_b,
                        mask_power=args.mask_power,
                    )
                    _, oracle_si_sdri = pit_two_source_si_sdr(
                        mix_wav=mix_wav[b, :length_b],
                        pred1_wav=oracle["source1_waveform"][0, :length_b],
                        pred2_wav=oracle["source2_waveform"][0, :length_b],
                        target1_wav=s1_wav[b, :length_b],
                        target2_wav=s2_wav[b, :length_b],
                    )
                    oracle_si_sdri_vals.append(oracle_si_sdri)

    if model_was_training:
        model.train()

    metrics = {
        "si_sdr_mean_db": float(sum(si_sdr_vals) / len(si_sdr_vals)) if si_sdr_vals else float("nan"),
        "si_sdri_mean_db": float(sum(si_sdri_vals) / len(si_sdri_vals)) if si_sdri_vals else float("nan"),
        "source_loss_mean": float(sum(source_loss_vals) / len(source_loss_vals)) if source_loss_vals else float("nan"),
        "mix_loss_mean": float(sum(mix_loss_vals) / len(mix_loss_vals)) if mix_loss_vals else float("nan"),
        "num_eval_examples": int(len(si_sdri_vals)),
    }
    if oracle_si_sdri_vals:
        oracle_mean = float(sum(oracle_si_sdri_vals) / len(oracle_si_sdri_vals))
        metrics["oracle_si_sdri_mean_db"] = oracle_mean
        metrics["oracle_gap_db"] = oracle_mean - metrics["si_sdri_mean_db"]
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_si_sdri: float,
    args: argparse.Namespace,
) -> None:
    state_dict = model.state_dict()
    payload = {
        # Save both keys for compatibility with the existing repo tooling.
        "state_dict": state_dict,
        "model_state": state_dict,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_si_sdri": best_si_sdri,
        "config": vars(args),
    }
    torch.save(payload, path)


def maybe_init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed; disabling wandb logging")
        return None

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config=vars(args),
    )


def train(args: argparse.Namespace) -> None:
    configure_logging()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_dataset = get_librimix_dataset(
        root_dir=args.librimix_root,
        split=args.train_split,
        sample_rate=args.sample_rate,
        mix_type=args.mix_type,
        mixture_dirname=args.mixture_dirname,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=args.max_frames,
        crop_mode=args.crop_mode,
    )
    train_loader = get_librimix_dataloader(
        root_dir=args.librimix_root,
        split=args.train_split,
        sample_rate=args.sample_rate,
        mix_type=args.mix_type,
        mixture_dirname=args.mixture_dirname,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=args.max_frames,
        crop_mode=args.crop_mode,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = get_librimix_dataloader(
        root_dir=args.librimix_root,
        split=args.val_split,
        sample_rate=args.sample_rate,
        mix_type=args.mix_type,
        mixture_dirname=args.mixture_dirname,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=None,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    input_channels, input_length = infer_input_shape(train_dataset)
    model = build_model(args, input_channels=input_channels, input_length=input_length).to(device)
    optimizer = build_optimizer(model, args)
    wandb_run = maybe_init_wandb(args)

    best_si_sdri = float("-inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_metrics: dict[str, float] = {}

        for batch in train_loader:
            mix_mag = batch["mixture_mag"].to(device)
            s1_mag = batch["source1_mag"].to(device)
            s2_mag = batch["source2_mag"].to(device)

            optimizer.zero_grad()
            out, aux = run_model(model, mix_mag, args, epoch=epoch, eval_mode=False)
            beta_frac = aux.get("beta_frac", 1.0)
            loss, metrics = compute_losses(
                out,
                mix_mag=mix_mag,
                source1_mag=s1_mag,
                source2_mag=s2_mag,
                args=args,
                device=device,
                beta_frac=beta_frac,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            for key, value in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + float(value)
            if "temp" in aux:
                epoch_metrics["temp"] = epoch_metrics.get("temp", 0.0) + float(aux["temp"])

        n_batches = len(train_loader)
        averaged = {key: value / n_batches for key, value in epoch_metrics.items()}
        averaged["epoch_runtime_sec"] = time.perf_counter() - t0
        averaged["epoch"] = epoch

        if epoch == 1 or epoch % args.log_every == 0:
            if args.experiment == "hybrid_partition":
                log.info(
                    "Epoch %4d | loss %.4f | source %.4f | mix %.4f | kl_g %.4f | kl_d %.4f | "
                    "wkl_g %.4f | wkl_d %.4f | n_act_frame %.1f | n_act_total %.1f/%d | "
                    "H(assign) %.3f | temp %.3f | %.1fs",
                    epoch,
                    averaged["total_loss"],
                    averaged["source_loss"],
                    averaged["mix_loss"],
                    averaged.get("kl_gamma", 0.0),
                    averaged.get("kl_delta", 0.0),
                    averaged.get("weighted_kl_gamma", 0.0),
                    averaged.get("weighted_kl_delta", 0.0),
                    averaged.get("n_active_frame", 0.0),
                    averaged.get("n_active_total", 0.0),
                    args.n_atoms,
                    averaged.get("assign_entropy", 0.0),
                    averaged.get("temp", args.temp_min),
                    averaged["epoch_runtime_sec"],
                )
            else:
                log.info(
                    "Epoch %4d | loss %.4f | source %.4f | %.1fs",
                    epoch,
                    averaged["total_loss"],
                    averaged["source_loss"],
                    averaged["epoch_runtime_sec"],
                )

        val_metrics = None
        if epoch == 1 or epoch % args.validate_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, device, args)
            log.info(
                "Validation epoch %4d | SI-SDRi %.3f dB | SI-SDR %.3f dB | source %.4f | mix %.4f | eval_n=%d",
                epoch,
                val_metrics["si_sdri_mean_db"],
                val_metrics["si_sdr_mean_db"],
                val_metrics["source_loss_mean"],
                val_metrics["mix_loss_mean"],
                val_metrics["num_eval_examples"],
            )
            if "oracle_gap_db" in val_metrics:
                log.info(
                    "Validation epoch %4d | oracle SI-SDRi %.3f dB | oracle gap %.3f dB",
                    epoch,
                    val_metrics["oracle_si_sdri_mean_db"],
                    val_metrics["oracle_gap_db"],
                )

            if val_metrics["si_sdri_mean_db"] > best_si_sdri:
                best_si_sdri = val_metrics["si_sdri_mean_db"]
                best_epoch = epoch
                save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_si_sdri, args)

        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_si_sdri, args)

        if wandb_run is not None:
            import wandb

            payload = {f"train/{k}": v for k, v in averaged.items()}
            if val_metrics is not None:
                payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(payload, step=epoch)

    summary = {
        "best_epoch": best_epoch,
        "best_si_sdri_mean_db": best_si_sdri,
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    log.info("Best validation SI-SDRi: %.3f dB at epoch %d", best_si_sdri, best_epoch)

    if wandb_run is not None:
        wandb_run.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LibriMix follow-up experiments")
    parser.add_argument("--experiment", type=str, required=True, choices=["direct_mask", "hybrid_partition"])
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--max-dev-eval", type=int, default=8)
    parser.add_argument("--compute-oracle-gap", action="store_true")
    parser.add_argument("--mask-power", type=float, default=2.0)

    parser.add_argument("--librimix-root", type=str, default="./data/Libri2Mix")
    parser.add_argument("--train-split", type=str, default="train-100")
    parser.add_argument("--val-split", type=str, default="dev")
    parser.add_argument("--mix-type", type=str, default="min")
    parser.add_argument("--mixture-dirname", type=str, default="mix_clean")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--crop-mode", type=str, default="center", choices=["center", "random"])

    parser.add_argument("--source-loss-type", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument("--mix-loss-type", type=str, default="mse", choices=["l1", "mse"])
    parser.add_argument("--lambda-source", type=float, default=1.0)
    parser.add_argument("--lambda-mix", type=float, default=1.0)

    parser.add_argument("--direct-hidden-channels", type=int, default=384)
    parser.add_argument("--direct-num-blocks", type=int, default=6)
    parser.add_argument("--direct-kernel-size", type=int, default=3)
    parser.add_argument("--direct-dropout", type=float, default=0.0)

    parser.add_argument("--encoder-type", type=str, default="resnet")
    parser.add_argument("--encoder-output-dim", type=int, default=256)
    parser.add_argument("--assignment-hidden", type=int, default=256)
    parser.add_argument("--n-atoms", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--motif-width", type=int, default=16)
    parser.add_argument("--decoder-stride", type=int, default=4)
    parser.add_argument("--match-encoder-decoder-stride", action="store_true")
    parser.add_argument("--dict-init", type=str, default="random")
    parser.add_argument("--no-normalize-dict", action="store_true")
    parser.add_argument("--dict-lr-mult", type=float, default=0.1)
    parser.add_argument("--magnitude-dist", type=str, default="gamma", choices=["gamma", "gaussian"])
    parser.add_argument("--structure-mode", type=str, default="binary", choices=["binary", "ternary"])
    parser.add_argument("--k-min", type=float, default=0.1)
    parser.add_argument("--k-max", type=float, default=0.8)
    parser.add_argument("--k0", type=float, default=0.3)
    parser.add_argument("--theta0", type=float, default=1.0)
    parser.add_argument("--delta-prior", type=str, default="0.98,0.02")
    parser.add_argument("--beta-gamma-final", type=float, default=0.005)
    parser.add_argument("--beta-delta-final", type=float, default=0.05)
    parser.add_argument("--kl-normalization", type=str, default="site", choices=["site", "batch"])
    parser.add_argument("--temp-init", type=float, default=1.0)
    parser.add_argument("--temp-min", type=float, default=0.05)
    parser.add_argument("--temp-anneal-epochs", type=int, default=50)
    parser.add_argument("--phase1-end", type=int, default=40)
    parser.add_argument("--phase2-end", type=int, default=60)
    parser.add_argument("--phase3-end", type=int, default=120)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="hybrid-sparse-vae")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
