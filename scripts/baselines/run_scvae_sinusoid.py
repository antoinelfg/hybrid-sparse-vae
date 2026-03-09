#!/usr/bin/env python
"""Train a sinusoid SC-VAE baseline with a linear LISTA-style encoder."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import generate_toy_sinusoid_tensors


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
    x, _, _, _ = generate_toy_sinusoid_tensors(
        n_samples=n_samples,
        length=length,
        n_components=n_components,
        seed=seed,
        gain_distribution=gain_distribution,
        gain_min=gain_min,
        gain_max=gain_max,
        normalize_divisor=normalize_divisor,
    )
    return TensorDataset(x)


def soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.relu(x.abs() - threshold)


class LinearLISTAEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 128,
        n_steps: int = 5,
        threshold_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.recurrent = nn.Linear(latent_dim, latent_dim, bias=False)
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)
        self.log_threshold = nn.Parameter(torch.full((latent_dim,), math.log(threshold_init)))
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drive = self.input_proj(x)
        threshold = self.log_threshold.exp().unsqueeze(0)
        z = soft_threshold(drive, threshold)
        for _ in range(self.n_steps - 1):
            z = soft_threshold(drive + self.recurrent(z), threshold)
        mu = self.mu_head(z)
        logvar = self.logvar_head(z).clamp(min=-8.0, max=4.0)
        return mu, logvar, z


class LinearDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).unsqueeze(1)


class LISTASCVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 128,
        n_steps: int = 5,
        prior_scale: float = 0.25,
        threshold_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = LinearLISTAEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_steps=n_steps,
            threshold_init=threshold_init,
        )
        self.decoder = LinearDecoder(latent_dim=latent_dim, output_dim=input_dim)
        self.prior_scale = prior_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        x_flat = x.view(x.size(0), -1)
        mu, logvar, lista_code = self.encoder(x_flat)
        std = (0.5 * logvar).exp()
        if self.training:
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        b = self.prior_scale
        var = logvar.exp()
        safe_std = std.clamp_min(1e-6)
        abs_mean = safe_std * math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * mu.pow(2) / var.clamp_min(1e-6))
        abs_mean = abs_mean + mu * torch.erf(mu / (safe_std * math.sqrt(2.0)))
        kl = (-0.5 * (1.0 + logvar) + math.log(2.0 * b) + abs_mean / b).sum(dim=-1).mean()

        recon = self.decoder(z)
        active = mu.abs() > self.encoder.log_threshold.exp().unsqueeze(0)
        return recon, {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "lista_code": lista_code,
            "kl": kl,
            "active": active,
            "threshold_mean": float(self.encoder.log_threshold.exp().mean().item()),
            "sparsity": float((~active).float().mean().item()),
        }


def train_model(
    model: LISTASCVAE,
    data: TensorDataset,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    beta: float,
    device: str,
) -> dict[str, float]:
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_recon = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        if epoch > epochs // 2:
            frac = (epoch - epochs // 2) / max(1, epochs // 2)
            lr_now = lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * frac))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, info = model(batch_x)
            recon_loss = F.mse_loss(recon, batch_x, reduction="sum") / batch_x.size(0)
            loss = recon_loss + beta * info["kl"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_recon += float(recon_loss.item())
            epoch_kl += float(info["kl"].item())

        avg_recon = epoch_recon / max(1, len(loader))
        best_recon = min(best_recon, avg_recon)
        if epoch % 200 == 0 or epoch == 1:
            print(
                f"[scvae_lista] Epoch {epoch:4d} | recon {avg_recon:.4f} | "
                f"kl {epoch_kl / max(1, len(loader)):.4f} | best {best_recon:.4f} | "
                f"sparse={info['sparsity']:.2%} | thr={info['threshold_mean']:.4f}"
            )

    model.eval()
    with torch.no_grad():
        all_x = data.tensors[0].to(device)
        recon, info = model(all_x)
        final_mse = float(F.mse_loss(recon, all_x, reduction="sum").item() / all_x.size(0))
        metrics = {
            "recon_mse": final_mse,
            "best_recon": best_recon,
            "sparsity": float(info["sparsity"]),
            "threshold_mean": float(info["threshold_mean"]),
            "n_active_mean": float(info["active"].float().sum(dim=1).mean().item()),
        }
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LISTA-style SC-VAE sinusoid baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--prior-scale", type=float, default=0.25)
    parser.add_argument("--threshold-init", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--gain-distribution", type=str, default="none")
    parser.add_argument("--gain-min", type=float, default=1.0)
    parser.add_argument("--gain-max", type=float, default=1.0)
    parser.add_argument("--normalize-divisor", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data = make_toy_data(
        n_samples=args.n_samples,
        length=args.length,
        n_components=args.n_components,
        seed=args.seed,
        gain_distribution=args.gain_distribution,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        normalize_divisor=args.normalize_divisor,
    )
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = LISTASCVAE(
        input_dim=args.length,
        latent_dim=args.latent_dim,
        n_steps=args.n_steps,
        prior_scale=args.prior_scale,
        threshold_init=args.threshold_init,
    )
    metrics = train_model(
        model,
        data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        beta=args.beta,
        device=device,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "state_dict": model.state_dict(),
            "config": {
                "seed": args.seed,
                "epochs": args.epochs,
                "latent_dim": args.latent_dim,
                "n_steps": args.n_steps,
                "prior_scale": args.prior_scale,
                "threshold_init": args.threshold_init,
                "beta": args.beta,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "n_samples": args.n_samples,
                "length": args.length,
                "n_components": args.n_components,
                "gain_distribution": args.gain_distribution,
                "gain_min": args.gain_min,
                "gain_max": args.gain_max,
                "normalize_divisor": args.normalize_divisor,
                "model_type": "scvae_lista",
            },
            "metrics": metrics,
        },
        args.checkpoint,
    )
    payload = {
        "seed": args.seed,
        "epochs": args.epochs,
        "length": args.length,
        "n_components": args.n_components,
        "n_samples": args.n_samples,
        "model": "scvae_lista",
        "results": metrics,
        "checkpoint": str(args.checkpoint),
        "gain_distribution": args.gain_distribution,
        "gain_min": args.gain_min,
        "gain_max": args.gain_max,
        "normalize_divisor": args.normalize_divisor,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
