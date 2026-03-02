#!/usr/bin/env python
"""Evaluate the generative quality of the trained model.

Metrics computed:
  1. MMD (Maximum Mean Discrepancy) — measures distribution distance.
  2. Coverage (Recall) — fraction of real signals with a generated neighbor within ε.
  3. Precision — fraction of generated signals with a real neighbor within ε.
  4. PSD Error — MSE between the mean Power Spectral Densities.

Usage:
  python scripts/evaluate_generation.py --checkpoint checkpoints/hybrid_vae_final.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributions as dist
import torch.nn.functional as F
import yaml

from scripts.visualize import build_model
from train import make_toy_data
from data.datasets import get_mnist_dataset


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = None) -> float:
    """Compute Maximum Mean Discrepancy (MMD) with a Gaussian kernel.
    
    x, y: [N, D] tensors
    """
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    if sigma is None:
        # Median heuristic
        sigma = torch.sqrt(xy.median()).item()
        if sigma == 0.0:
            sigma = 1.0

    k_xx = torch.exp(-xx / (2 * sigma**2)).mean()
    k_yy = torch.exp(-yy / (2 * sigma**2)).mean()
    k_xy = torch.exp(-xy / (2 * sigma**2)).mean()

    return (k_xx + k_yy - 2 * k_xy).item()


def compute_precision_recall(real: torch.Tensor, gen: torch.Tensor, threshold: float = 0.5):
    """Compute precision and recall based on 1-NN distances.
    
    Recall (Coverage): % of real samples covered by a gen sample.
    Precision: % of gen samples close to a real sample.
    """
    # pairwise distances [N_real, N_gen]
    dists = torch.cdist(real, gen, p=2)
    
    # Coverage: for each real, min distance to a gen
    min_dist_to_gen, _ = dists.min(dim=1)
    recall = (min_dist_to_gen < threshold).float().mean().item()
    mean_dist_to_gen = min_dist_to_gen.mean().item()
    
    # Precision: for each gen, min distance to a real
    min_dist_to_real, _ = dists.min(dim=0)
    precision = (min_dist_to_real < threshold).float().mean().item()
    mean_dist_to_real = min_dist_to_real.mean().item()
    
    return precision, recall, mean_dist_to_real, mean_dist_to_gen


def compute_psd_error(real: torch.Tensor, gen: torch.Tensor) -> float:
    """Compare average Power Spectral Density (PSD)."""
    # real, gen: [..., T]
    fft_real = torch.fft.rfft(real, dim=-1)
    psd_real = torch.abs(fft_real).pow(2).mean(dim=0)  # Average over batch
    
    fft_gen = torch.fft.rfft(gen, dim=-1)
    psd_gen = torch.abs(fft_gen).pow(2).mean(dim=0)
    
    # Normalize
    psd_real = psd_real / psd_real.sum().clamp_min(1e-12)
    psd_gen = psd_gen / psd_gen.sum().clamp_min(1e-12)
    
    return F.mse_loss(psd_gen, psd_real).item()


def find_hydra_config(ckpt_path: Path) -> Path | None:
    """Find nearest .hydra/config.yaml from checkpoint parent upward."""
    search_roots = [ckpt_path.parent, *ckpt_path.parents]
    for root in search_roots:
        cfg_path = root / ".hydra" / "config.yaml"
        if cfg_path.exists():
            return cfg_path
        if root == REPO_ROOT:
            break
    return None


def load_hydra_config(ckpt_path: Path) -> tuple[dict[str, Any], Path | None]:
    """Load Hydra config dict if available near the checkpoint."""
    cfg_path = find_hydra_config(ckpt_path)
    if cfg_path is None:
        return {}, None

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - defensive, CLI side path
        print(f"WARNING: Could not parse Hydra config at {cfg_path}: {exc}")
        return {}, None

    if not isinstance(data, dict):
        print(f"WARNING: Hydra config at {cfg_path} is not a dict. Ignoring.")
        return {}, None
    return data, cfg_path


def parse_delta_prior(delta_prior: str, expected_len: int, device: torch.device) -> torch.Tensor:
    """Parse and validate comma-separated prior probabilities."""
    parts = [p.strip() for p in delta_prior.split(",") if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid --delta-prior value: {delta_prior}") from exc

    if len(values) != expected_len:
        raise ValueError(
            f"--delta-prior has {len(values)} values, expected {expected_len} "
            f"for structure_mode='{expected_len == 3 and 'ternary' or 'binary'}'."
        )

    probs = torch.tensor(values, device=device, dtype=torch.float32)
    if torch.any(probs < 0):
        raise ValueError("--delta-prior must contain non-negative values.")

    total = probs.sum().item()
    if total <= 0:
        raise ValueError("--delta-prior probabilities must sum to a positive value.")
    return probs / total


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load plain or wrapped state_dict from checkpoint."""
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Backward compatibility with older PyTorch versions without weights_only.
        payload = torch.load(path, map_location=device)
    except Exception as exc:
        # Some .ckpt files may include metadata blocked by weights_only loading.
        print(f"WARNING: weights_only checkpoint loading failed ({exc}). Retrying full load.")
        payload = torch.load(path, map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")

    if not state_dict:
        raise ValueError("Checkpoint state_dict is empty.")

    # Common wrapper prefix (e.g., Lightning)
    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate Generative Quality")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="sinusoid", choices=["sinusoid", "mnist"])
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Model config overrides. If omitted, the script tries to read .hydra/config.yaml
    # next to the checkpoint before falling back to conservative defaults.
    parser.add_argument("--dict-init", type=str, default=None)
    parser.add_argument("--n-atoms", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--encoder-output-dim", type=int, default=None)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--encoder-type", type=str, default=None)
    parser.add_argument("--decoder-type", type=str, default=None)
    parser.add_argument("--k-min", type=float, default=None)
    parser.add_argument("--magnitude-dist", type=str, default=None, choices=["gamma", "gaussian"])
    parser.add_argument("--structure-mode", type=str, default=None, choices=["ternary", "binary"])
    
    # Prior config
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--theta0", type=float, default=None)
    parser.add_argument("--delta-prior", type=str, default=None)
    
    # Threshold for P/R
    parser.add_argument("--threshold", type=float, default=15.0)

    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        sys.exit(1)
        
    hydra_cfg, hydra_cfg_path = load_hydra_config(ckpt_path)
    if hydra_cfg_path is not None:
        print(f"Using Hydra config defaults from: {hydra_cfg_path}")

    def resolve_arg(arg_name: str, default: Any, hydra_key: str | None = None) -> Any:
        explicit = getattr(args, arg_name)
        if explicit is not None:
            return explicit
        key = hydra_key if hydra_key is not None else arg_name
        return hydra_cfg.get(key, default)

    # Resolve model config
    args.dict_init = resolve_arg("dict_init", "dct")
    args.n_atoms = int(resolve_arg("n_atoms", 128))
    args.latent_dim = int(resolve_arg("latent_dim", 64))
    args.encoder_output_dim = int(resolve_arg("encoder_output_dim", 256))
    args.input_length = int(resolve_arg("input_length", 128))
    args.encoder_type = resolve_arg("encoder_type", "linear")
    args.decoder_type = resolve_arg("decoder_type", "linear")
    args.k_min = float(resolve_arg("k_min", 0.1))
    args.magnitude_dist = resolve_arg("magnitude_dist", "gamma")
    args.structure_mode = resolve_arg("structure_mode", "ternary")

    # Resolve prior config
    args.k0 = float(resolve_arg("k0", 0.3, hydra_key="k_0"))
    args.theta0 = float(resolve_arg("theta0", 1.0, hydra_key="theta_0"))
    delta_prior_raw = resolve_arg("delta_prior", "0.15,0.70,0.15")
    if isinstance(delta_prior_raw, (list, tuple)):
        args.delta_prior = ",".join(str(float(x)) for x in delta_prior_raw)
    else:
        args.delta_prior = str(delta_prior_raw)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {ckpt_path.name}...")
    model = build_model(args)
    state_dict = load_state_dict(ckpt_path, device=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # ---- 1. Real Data ------------------------------------------
    if args.dataset == "sinusoid":
        dataset = make_toy_data(
            n_samples=args.n_samples,
            length=args.input_length,
            n_components=3,
            seed=42,
        )
        X_real = dataset.tensors[0].to(device)  # [N, 1, T]
    elif args.dataset == "mnist":
        dataset = get_mnist_dataset(data_dir=str(REPO_ROOT / "data"), flatten=True)
        # Limit to n_samples to match generative size
        batch_size = min(args.n_samples, len(dataset))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch = next(iter(loader))
        X_real = batch[0] if isinstance(batch, (tuple, list)) else batch
        X_real = X_real.to(device)  # [N, 1, T] depending on internal shape
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    n_eval = X_real.shape[0]
    if n_eval != args.n_samples:
        print(f"Requested {args.n_samples} samples, using {n_eval} available real samples.")
    
    # ---- 2. Generated Data -------------------------------------
    expected_probs = 3 if args.structure_mode == "ternary" else 2
    probs = parse_delta_prior(args.delta_prior, expected_probs, device=device)
    n_atoms = model.latent.n_atoms

    if args.dataset == "mnist" and args.magnitude_dist == "gamma" and args.k0 < 5.0:
        print(
            f"WARNING: k0={args.k0} may be too small for dense MNIST checkpoints. "
            "If this is a 'safe' model, pass the training prior (e.g. --k0 40.0)."
        )
    
    with torch.no_grad():
        if args.magnitude_dist == "gaussian":
            gamma = torch.randn((n_eval, n_atoms), device=device)
        else:
            gamma = dist.Gamma(
                torch.full((n_eval, n_atoms), args.k0, device=device),
                torch.full((n_eval, n_atoms), 1.0 / args.theta0, device=device),
            ).sample()

        cat_idx = dist.Categorical(probs=probs).sample((n_eval, n_atoms))
        
        # Mapping for ternary or binary
        if args.structure_mode == "ternary":
            delta_map = torch.tensor([-1.0, 0.0, 1.0], device=device)
        else:
            delta_map = torch.tensor([0.0, 1.0], device=device)
            
        delta = delta_map[cat_idx]
        
        B = gamma * delta
        z = model.latent.dictionary(B)
        X_gen = model.decoder(z)
        
    # ---- 3. Compute Metrics ------------------------------------
    print(f"Processing {n_eval} samples...")
    
    # Flatten [N, 1, T] -> [N, T]
    real_flat = X_real.reshape(n_eval, -1)
    gen_flat = X_gen.reshape(n_eval, -1)
    
    mmd_val = compute_mmd(real_flat, gen_flat, sigma=None)
    precision, recall, mean_dist_to_real, mean_dist_to_gen = compute_precision_recall(real_flat, gen_flat, threshold=args.threshold)
    psd_err = compute_psd_error(real_flat, gen_flat)
    
    print("\n" + "=" * 40)
    print(" GENERATIVE QUALITY METRICS ")
    print("=" * 40)
    print(f" MMD (median heuristic) : {mmd_val:.4f}  (lower is better)")
    print(f" PSD Error (MSE) : {psd_err:.6f}  (lower is better)")
    print(f" Precision (t={args.threshold}) : {precision:.2%}  (% gen realistic)")
    print(f"   Avg dist to real    : {mean_dist_to_real:.4f}")
    print(f" Recall/Cov (t={args.threshold}) : {recall:.2%}  (% real covered)")
    print(f"   Avg dist to gen     : {mean_dist_to_gen:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
