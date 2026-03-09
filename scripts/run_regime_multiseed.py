#!/usr/bin/env python
"""Run the structured sparse regime-study conditions across multiple seeds."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [42, 123, 456, 789, 1337]

HSVAE_COMMON = [
    "dataset=sinusoid",
    "input_length=128",
    "batch_size=256",
    "encoder_type=linear",
    "decoder_type=linear",
    "n_atoms=128",
    "latent_dim=64",
    "encoder_output_dim=256",
    "dict_init=random",
    "normalize_dict=True",
    "structure_mode=ternary",
    "use_wandb=False",
]

CONDITIONS = {
    "HSVAE-lowk-full": {
        "kind": "hsvae",
        "overrides": [
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=site",
        ],
    },
    "HSVAE-lowk-oldkl": {
        "kind": "hsvae",
        "overrides": [
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-highk-safe": {
        "kind": "hsvae",
        "overrides": [
            "k_min=10.0",
            "k_max=50.0",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=site",
        ],
    },
    "HSVAE-no-delta": {
        "kind": "hsvae",
        "overrides": [
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.0",
            "kl_normalization=site",
        ],
    },
    "HSVAE-lowk-lista": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=lista",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-lowk-lista-lat128": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=lista",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "latent_dim=128",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-lowk-lista-delta-l2": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=lista",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "delta_head_mode=l2norm",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-lowk-lista-lat128-delta-l2": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=lista",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "latent_dim=128",
            "delta_head_mode=l2norm",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-polar-lista-gumbel": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=polar_lista",
            "polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "shape_norm=l2_global",
            "gain_feature=log_l2",
            "shape_detach_to_gamma=True",
            "delta_factorization=presence_sign",
            "presence_estimator=gumbel_binary",
            "tau_presence_eval=0.5",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-polar-lista-sparsemax": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=polar_lista",
            "polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "shape_norm=l2_global",
            "gain_feature=log_l2",
            "shape_detach_to_gamma=True",
            "delta_factorization=presence_sign",
            "presence_estimator=sparsemax_binary",
            "tau_presence_eval=0.5",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-polar-lista-entmax15": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=polar_lista",
            "polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "shape_norm=l2_global",
            "gain_feature=log_l2",
            "shape_detach_to_gamma=True",
            "delta_factorization=presence_sign",
            "presence_estimator=entmax15_binary",
            "presence_alpha=1.5",
            "tau_presence_eval=0.5",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-fully-polar-lista-gumbel": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=fully_polar_lista",
            "fully_polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "delta_factorization=presence_sign",
            "presence_estimator=gumbel_binary",
            "sign_estimator=gumbel_binary",
            "tau_presence_eval=0.5",
            "sign_tau_eval=0.5",
            "shape_norm=l2_global",
            "gamma_scale_injection=multiply_input_norm",
            "shape_detach_to_gamma=True",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.0",
            "beta_presence_final=0.1",
            "beta_sign_final=0.02",
            "presence_prior='0.90,0.10'",
            "sign_prior='0.50,0.50'",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-fully-polar-lista-gumbel-energy-div4": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=fully_polar_lista",
            "fully_polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "delta_factorization=presence_sign",
            "presence_estimator=gumbel_binary",
            "sign_estimator=gumbel_binary",
            "tau_presence_eval=0.5",
            "sign_tau_eval=0.5",
            "shape_norm=l2_global",
            "gamma_scale_injection=multiply_input_norm",
            "shape_detach_to_gamma=True",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.0",
            "beta_presence_final=0.1",
            "beta_sign_final=0.02",
            "presence_prior='0.90,0.10'",
            "sign_prior='0.50,0.50'",
            "kl_normalization=batch",
            "sinusoid_gain_distribution=log_uniform",
            "sinusoid_gain_min=0.1",
            "sinusoid_gain_max=10.0",
            "sinusoid_normalize_divisor=4.0",
        ],
    },
    "HSVAE-fully-polar-lista-sparsemax": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=fully_polar_lista",
            "fully_polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "delta_factorization=presence_sign",
            "presence_estimator=sparsemax_binary",
            "sign_estimator=gumbel_binary",
            "tau_presence_eval=0.5",
            "sign_tau_eval=0.5",
            "shape_norm=l2_global",
            "gamma_scale_injection=multiply_input_norm",
            "shape_detach_to_gamma=True",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.0",
            "beta_presence_final=0.1",
            "beta_sign_final=0.02",
            "presence_prior='0.90,0.10'",
            "sign_prior='0.50,0.50'",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-fully-polar-lista-sparsemax-energy-div4": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=fully_polar_lista",
            "fully_polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "delta_factorization=presence_sign",
            "presence_estimator=sparsemax_binary",
            "sign_estimator=gumbel_binary",
            "tau_presence_eval=0.5",
            "sign_tau_eval=0.5",
            "shape_norm=l2_global",
            "gamma_scale_injection=multiply_input_norm",
            "shape_detach_to_gamma=True",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.0",
            "beta_presence_final=0.1",
            "beta_sign_final=0.02",
            "presence_prior='0.90,0.10'",
            "sign_prior='0.50,0.50'",
            "kl_normalization=batch",
            "sinusoid_gain_distribution=log_uniform",
            "sinusoid_gain_min=0.1",
            "sinusoid_gain_max=10.0",
            "sinusoid_normalize_divisor=4.0",
        ],
    },
    "HSVAE-fully-polar-lista-entmax15": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=fully_polar_lista",
            "fully_polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "delta_factorization=presence_sign",
            "presence_estimator=entmax15_binary",
            "presence_alpha=1.5",
            "sign_estimator=gumbel_binary",
            "tau_presence_eval=0.5",
            "sign_tau_eval=0.5",
            "shape_norm=l2_global",
            "gamma_scale_injection=multiply_input_norm",
            "shape_detach_to_gamma=True",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.0",
            "beta_presence_final=0.1",
            "beta_sign_final=0.02",
            "presence_prior='0.90,0.10'",
            "sign_prior='0.50,0.50'",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-fully-polar-lista-entmax15-energy-div4": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=fully_polar_lista",
            "fully_polar_encoder=True",
            "latent_dim=128",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "delta_factorization=presence_sign",
            "presence_estimator=entmax15_binary",
            "presence_alpha=1.5",
            "sign_estimator=gumbel_binary",
            "tau_presence_eval=0.5",
            "sign_tau_eval=0.5",
            "shape_norm=l2_global",
            "gamma_scale_injection=multiply_input_norm",
            "shape_detach_to_gamma=True",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.0",
            "beta_presence_final=0.1",
            "beta_sign_final=0.02",
            "presence_prior='0.90,0.10'",
            "sign_prior='0.50,0.50'",
            "kl_normalization=batch",
            "sinusoid_gain_distribution=log_uniform",
            "sinusoid_gain_min=0.1",
            "sinusoid_gain_max=10.0",
            "sinusoid_normalize_divisor=4.0",
        ],
    },
    "HSVAE-lowk-lista-energy": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=lista",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
            "sinusoid_gain_distribution=log_uniform",
            "sinusoid_gain_min=0.1",
            "sinusoid_gain_max=10.0",
            "sinusoid_normalize_divisor=40.0",
        ],
    },
    "HSVAE-lowk-lista-energy-div4": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=lista",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
            "sinusoid_gain_distribution=log_uniform",
            "sinusoid_gain_min=0.1",
            "sinusoid_gain_max=10.0",
            "sinusoid_normalize_divisor=4.0",
        ],
    },
    "HSVAE-lowk-lista-tightdelta": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=lista",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "gumbel_epsilon=0.35",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
        ],
    },
    "HSVAE-lowk-lista-softft": {
        "kind": "hsvae",
        "overrides": [
            "encoder_type=lista",
            "lista_iterations=5",
            "lista_threshold_init=0.1",
            "k_min=0.01",
            "k_max=0.8",
            "beta_gamma_final=0.005",
            "beta_delta_final=0.1",
            "kl_normalization=batch",
            "soft_finetune_start=2500",
        ],
    },
    "SimpleSparseBaseline": {
        "kind": "baseline",
    },
    "SCVAE-lista": {
        "kind": "scvae",
    },
    "SCVAE-lista-energy": {
        "kind": "scvae",
        "cli_args": [
            "--gain-distribution",
            "log_uniform",
            "--gain-min",
            "0.1",
            "--gain-max",
            "10.0",
            "--normalize-divisor",
            "40.0",
        ],
    },
    "SCVAE-lista-energy-div4": {
        "kind": "scvae",
        "cli_args": [
            "--gain-distribution",
            "log_uniform",
            "--gain-min",
            "0.1",
            "--gain-max",
            "10.0",
            "--normalize-divisor",
            "4.0",
        ],
    },
    "OMP-energy": {
        "kind": "omp_ref",
        "cli_args": [
            "--n-samples",
            "512",
            "--length",
            "128",
            "--n-components",
            "3",
            "--gain-distribution",
            "log_uniform",
            "--gain-min",
            "0.1",
            "--gain-max",
            "10.0",
            "--normalize-divisor",
            "40.0",
            "--n-atoms",
            "128",
            "--n-nonzero",
            "33",
        ],
    },
    "OMP-energy-div4": {
        "kind": "omp_ref",
        "cli_args": [
            "--n-samples",
            "512",
            "--length",
            "128",
            "--n-components",
            "3",
            "--gain-distribution",
            "log_uniform",
            "--gain-min",
            "0.1",
            "--gain-max",
            "10.0",
            "--normalize-divisor",
            "4.0",
            "--n-atoms",
            "128",
            "--n-nonzero",
            "33",
        ],
    },
}


def _fully_polar_softprior_base(*, energy_div4: bool = False) -> list[str]:
    overrides = [
        "encoder_type=fully_polar_lista",
        "fully_polar_encoder=True",
        "latent_dim=128",
        "lista_iterations=5",
        "lista_threshold_init=0.1",
        "delta_factorization=presence_sign",
        "sign_estimator=gumbel_binary",
        "tau_presence_eval=0.5",
        "sign_tau_eval=0.5",
        "shape_norm=l2_global",
        "gamma_scale_injection=multiply_input_norm",
        "shape_detach_to_gamma=True",
        "k_min=0.01",
        "k_max=0.8",
        "beta_gamma_final=0.005",
        "beta_delta_final=0.0",
        "beta_sign_final=0.02",
        "sign_prior='0.50,0.50'",
        "kl_normalization=batch",
        "phase1_end=100",
        "phase2_end=300",
        "phase3_end=700",
        "temp_min=0.2",
        "temp_anneal_epochs=800",
        "sinusoid_sparse_monitor_every=100",
        "sinusoid_sparse_monitor_samples=256",
    ]
    if energy_div4:
        overrides.extend(
            [
                "sinusoid_gain_distribution=log_uniform",
                "sinusoid_gain_min=0.1",
                "sinusoid_gain_max=10.0",
                "sinusoid_normalize_divisor=4.0",
            ]
        )
    return overrides


def _fully_polar_p20_bp020_base(*, energy_div4: bool = False) -> list[str]:
    return _fully_polar_softprior_base(energy_div4=energy_div4) + [
        "presence_estimator=gumbel_binary",
        "beta_presence_final=0.02",
        "presence_prior='0.80,0.20'",
    ]


def _register_fully_polar_presence_sweep() -> None:
    standard_base = _fully_polar_softprior_base()
    energy_base = _fully_polar_softprior_base(energy_div4=True)

    for active_prob in (0.15, 0.20, 0.25, 0.30):
        absent_prob = 1.0 - active_prob
        for beta_presence in (0.01, 0.02, 0.03, 0.05):
            condition_name = (
                "HSVAE-fully-polar-lista-softprior-presweep-"
                f"p{int(round(active_prob * 100)):02d}-bp{int(round(beta_presence * 1000)):03d}"
            )
            CONDITIONS[condition_name] = {
                "kind": "hsvae",
                "overrides": standard_base + [
                    "presence_estimator=gumbel_binary",
                    f"beta_presence_final={beta_presence}",
                    f"presence_prior='{absent_prob:.2f},{active_prob:.2f}'",
                ],
            }

    estimator_settings = {
        "gumbel_binary": [],
        "sparsemax_binary": [],
        "entmax15_binary": ["presence_alpha=1.5"],
    }
    for estimator, extra in estimator_settings.items():
        short_name = estimator.replace("_binary", "")
        standard_name = f"HSVAE-fully-polar-lista-softprior-estimator-{short_name}"
        energy_name = f"HSVAE-fully-polar-lista-softprior-energy-div4-estimator-{short_name}"
        common_overrides = [
            f"presence_estimator={estimator}",
            "beta_presence_final=0.03",
            "presence_prior='0.80,0.20'",
        ] + extra
        CONDITIONS[standard_name] = {
            "kind": "hsvae",
            "overrides": standard_base + common_overrides,
        }
        CONDITIONS[energy_name] = {
            "kind": "hsvae",
            "overrides": energy_base + common_overrides,
        }

        bestpoint_standard_name = f"HSVAE-fully-polar-lista-softprior-p20-bp020-estimator-{short_name}"
        bestpoint_energy_name = f"HSVAE-fully-polar-lista-softprior-energy-div4-p20-bp020-estimator-{short_name}"
        bestpoint_overrides = [
            f"presence_estimator={estimator}",
            "beta_presence_final=0.02",
            "presence_prior='0.80,0.20'",
        ] + extra
        CONDITIONS[bestpoint_standard_name] = {
            "kind": "hsvae",
            "overrides": standard_base + bestpoint_overrides,
        }
        CONDITIONS[bestpoint_energy_name] = {
            "kind": "hsvae",
            "overrides": energy_base + bestpoint_overrides,
        }


def _register_fully_polar_dict_sweep() -> None:
    variants = [
        ("HSVAE-fully-polar-lista-softprior-p20-bp020", _fully_polar_p20_bp020_base()),
        (
            "HSVAE-fully-polar-lista-softprior-energy-div4-p20-bp020",
            _fully_polar_p20_bp020_base(energy_div4=True),
        ),
    ]
    for prefix, base in variants:
        for dict_init in ("random", "dct", "identity"):
            for dict_lr_mult, lr_tag in ((0.1, "010"), (1.0, "100"), (5.0, "500")):
                condition_name = f"{prefix}-dictinit-{dict_init}-dictlr-{lr_tag}"
                CONDITIONS[condition_name] = {
                    "kind": "hsvae",
                    "overrides": base + [
                        f"dict_init={dict_init}",
                        f"dict_lr_mult={dict_lr_mult}",
                    ],
                }


def _register_fully_polar_objective_sweep() -> None:
    variants = [
        (
            "HSVAE-fully-polar-lista-softprior-p20-bp020-dictinit-identity-dictlr-010",
            _fully_polar_p20_bp020_base() + [
                "dict_init=identity",
                "dict_lr_mult=0.1",
            ],
        ),
        (
            "HSVAE-fully-polar-lista-softprior-energy-div4-p20-bp020-dictinit-identity-dictlr-010",
            _fully_polar_p20_bp020_base(energy_div4=True) + [
                "dict_init=identity",
                "dict_lr_mult=0.1",
            ],
        ),
    ]
    for prefix, base in variants:
        for gamma_kl_target, gamma_tag in (("theta_tilde", "tilde"), ("theta_final", "final")):
            for lambda_presence_consistency, pcons_tag in (
                (0.0, "000"),
                (0.01, "010"),
                (0.02, "020"),
                (0.05, "050"),
            ):
                condition_name = f"{prefix}-gammakl-{gamma_tag}-pcons-{pcons_tag}"
                CONDITIONS[condition_name] = {
                    "kind": "hsvae",
                    "overrides": base + [
                        f"gamma_kl_target={gamma_kl_target}",
                        f"lambda_presence_consistency_final={lambda_presence_consistency}",
                        "presence_consistency_target=argmax",
                    ],
                }


def _extract_condition_override(condition: dict[str, object], key: str) -> str | None:
    for override in condition.get("overrides", []):
        if not isinstance(override, str):
            continue
        prefix = f"{key}="
        if override.startswith(prefix):
            value = override[len(prefix):]
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                return value[1:-1]
            return value
    return None


_register_fully_polar_presence_sweep()
_register_fully_polar_dict_sweep()
_register_fully_polar_objective_sweep()


def run_command(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def aggregate_numeric(values: list[float]) -> dict[str, float]:
    import math

    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {"mean": mean, "std": std, "min": min(values), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured sparse regime-study experiments across seeds")
    parser.add_argument(
        "--conditions",
        type=str,
        default="all",
        help="Comma-separated condition names or 'all'",
    )
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--baseline-epochs", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/regime_study")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.conditions == "all":
        conditions = list(CONDITIONS.keys())
    else:
        conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    summary_rows = []
    for condition_name in conditions:
        condition = CONDITIONS[condition_name]
        condition_dir = args.output_root / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        per_seed_metrics = []

        for seed in seeds:
            run_dir = condition_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            if condition["kind"] == "hsvae":
                output_json = run_dir / "sparse_recovery.json"
                checkpoint_path = run_dir / "hybrid_vae_final.pt"
                if args.skip_existing and output_json.exists():
                    metrics = json.loads(output_json.read_text(encoding="utf-8"))
                    per_seed_metrics.append(metrics)
                    continue

                train_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "train.py"),
                    f"seed={seed}",
                    f"epochs={args.epochs}",
                    f"device={args.device}",
                    f"save_dir={run_dir}",
                    f"hydra.run.dir={run_dir}",
                ]
                train_cmd.extend(HSVAE_COMMON)
                train_cmd.extend(condition["overrides"])
                run_command(train_cmd, cwd=REPO_ROOT)

                eval_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/eval_sparse_recovery.py"),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--hydra-config",
                    str(run_dir / ".hydra/config.yaml"),
                    "--output-json",
                    str(output_json),
                ]
                run_command(eval_cmd, cwd=REPO_ROOT)
                metrics = json.loads(output_json.read_text(encoding="utf-8"))
                per_seed_metrics.append(metrics)
                continue

            if condition["kind"] == "scvae":
                output_json = run_dir / "scvae_metrics.json"
                checkpoint_path = run_dir / "scvae_lista.pt"
                sparse_eval_json = run_dir / "sparse_recovery.json"
                if args.skip_existing and sparse_eval_json.exists():
                    metrics = json.loads(sparse_eval_json.read_text(encoding="utf-8"))
                    per_seed_metrics.append(metrics)
                    continue

                if not (args.skip_existing and output_json.exists() and checkpoint_path.exists()):
                    scvae_cmd = [
                        sys.executable,
                        str(REPO_ROOT / "scripts/baselines/run_scvae_sinusoid.py"),
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.baseline_epochs),
                        "--device",
                        args.device,
                        "--output-json",
                        str(output_json),
                        "--checkpoint",
                        str(checkpoint_path),
                    ]
                    scvae_cmd.extend(condition.get("cli_args", []))
                    run_command(scvae_cmd, cwd=REPO_ROOT)

                eval_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/eval_scvae_sparse_recovery.py"),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--seed",
                    "0",
                    "--output-json",
                    str(sparse_eval_json),
                ]
                run_command(eval_cmd, cwd=REPO_ROOT)
                metrics = json.loads(sparse_eval_json.read_text(encoding="utf-8"))
                per_seed_metrics.append(metrics)
                continue

            if condition["kind"] == "omp_ref":
                sparse_eval_json = run_dir / "sparse_recovery.json"
                if args.skip_existing and sparse_eval_json.exists():
                    metrics = json.loads(sparse_eval_json.read_text(encoding="utf-8"))
                    per_seed_metrics.append(metrics)
                    continue

                omp_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/eval_omp_sparse_recovery.py"),
                    "--seed",
                    str(seed),
                    "--output-json",
                    str(sparse_eval_json),
                ]
                omp_cmd.extend(condition.get("cli_args", []))
                run_command(omp_cmd, cwd=REPO_ROOT)
                metrics = json.loads(sparse_eval_json.read_text(encoding="utf-8"))
                per_seed_metrics.append(metrics)
                continue

            output_json = run_dir / "baseline_metrics.json"
            sparse_eval_json = run_dir / "baseline_sparse_recovery.json"
            if args.skip_existing and output_json.exists():
                payload = json.loads(output_json.read_text(encoding="utf-8"))
            else:
                baseline_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/baselines/run_baselines.py"),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.baseline_epochs),
                    "--device",
                    args.device,
                    "--output-json",
                    str(output_json),
                ]
                run_command(baseline_cmd, cwd=REPO_ROOT)
                payload = json.loads(output_json.read_text(encoding="utf-8"))
            if not (args.skip_existing and sparse_eval_json.exists()):
                sparse_eval_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/eval_sparse_baselines.py"),
                    "--baseline-json",
                    str(output_json),
                    "--output-json",
                    str(sparse_eval_json),
                ]
                run_command(sparse_eval_cmd, cwd=REPO_ROOT)
            sparse_eval_payload = json.loads(sparse_eval_json.read_text(encoding="utf-8"))
            selected_key = sparse_eval_payload.get("selected_baseline", "omp_33")
            per_seed_metrics.append(sparse_eval_payload["baselines"][selected_key])

        recon_values = [m["reconstruction"]["recon_mse_per_example"] for m in per_seed_metrics]
        support_values = [m.get("support_eval", {}).get("support_f1_mean") for m in per_seed_metrics]
        support_values = [value for value in support_values if value is not None]
        collapse_values = [1.0 if m["latents"]["collapsed"] else 0.0 for m in per_seed_metrics]
        row = {
            "condition": condition_name,
            "n_seeds": len(per_seed_metrics),
            "presence_estimator": _extract_condition_override(condition, "presence_estimator"),
            "sign_estimator": _extract_condition_override(condition, "sign_estimator"),
            "presence_prior": _extract_condition_override(condition, "presence_prior"),
            "sign_prior": _extract_condition_override(condition, "sign_prior"),
            "beta_gamma_final": _extract_condition_override(condition, "beta_gamma_final"),
            "beta_delta_final": _extract_condition_override(condition, "beta_delta_final"),
            "beta_presence_final": _extract_condition_override(condition, "beta_presence_final"),
            "beta_sign_final": _extract_condition_override(condition, "beta_sign_final"),
            "lambda_presence_consistency_final": _extract_condition_override(condition, "lambda_presence_consistency_final"),
            "gamma_kl_target": _extract_condition_override(condition, "gamma_kl_target"),
            "presence_consistency_target": _extract_condition_override(condition, "presence_consistency_target"),
            "phase1_end": _extract_condition_override(condition, "phase1_end"),
            "phase2_end": _extract_condition_override(condition, "phase2_end"),
            "phase3_end": _extract_condition_override(condition, "phase3_end"),
            "temp_init": _extract_condition_override(condition, "temp_init"),
            "temp_min": _extract_condition_override(condition, "temp_min"),
            "temp_anneal_epochs": _extract_condition_override(condition, "temp_anneal_epochs"),
            "dict_init": _extract_condition_override(condition, "dict_init"),
            "dict_lr_mult": _extract_condition_override(condition, "dict_lr_mult"),
            "gain_distribution": _extract_condition_override(condition, "sinusoid_gain_distribution"),
            "gain_min": _extract_condition_override(condition, "sinusoid_gain_min"),
            "gain_max": _extract_condition_override(condition, "sinusoid_gain_max"),
            "normalize_divisor": _extract_condition_override(condition, "sinusoid_normalize_divisor"),
            "recon_mse_mean": aggregate_numeric(recon_values)["mean"],
            "recon_mse_std": aggregate_numeric(recon_values)["std"],
            "support_f1_mean": aggregate_numeric(support_values)["mean"] if support_values else None,
            "support_f1_std": aggregate_numeric(support_values)["std"] if support_values else None,
            "collapse_rate": aggregate_numeric(collapse_values)["mean"],
        }
        summary_rows.append(row)

        with (condition_dir / "aggregate.json").open("w", encoding="utf-8") as handle:
            json.dump({"condition": condition_name, "seeds": seeds, "runs": per_seed_metrics, "summary": row}, handle, indent=2)

    summary_csv = args.output_root / "aggregate.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote multiseed summary to {summary_csv}")


if __name__ == "__main__":
    main()
