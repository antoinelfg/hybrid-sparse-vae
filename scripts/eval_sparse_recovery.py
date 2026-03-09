#!/usr/bin/env python
"""Evaluate controlled sparse recovery on toy sinusoid checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sinusoid_recovery import (
    SinusoidRecoverySpec,
    build_fourier_bank,
    build_fourier_subspaces,
    collapse_flag,
    generate_sinusoid_recovery_batch,
    support_metrics_from_scores,
)
from modules.latent_space import (
    binary_presence_projection,
    binary_sign_projection,
    combine_presence_sign,
    combine_presence_and_sign,
)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    if all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def read_hydra_config(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_k_min_from_hydra(path: Path | None) -> float | None:
    cfg = read_hydra_config(path)
    if cfg is None:
        return None
    value = cfg.get("k_min")
    return float(value) if value is not None else None


def read_sinusoid_stress_from_hydra(path: Path | None) -> dict[str, object]:
    cfg = read_hydra_config(path) or {}
    return {
        "gain_distribution": cfg.get("sinusoid_gain_distribution"),
        "gain_min": cfg.get("sinusoid_gain_min"),
        "gain_max": cfg.get("sinusoid_gain_max"),
        "normalize_divisor": cfg.get("sinusoid_normalize_divisor"),
    }


def read_lista_iterations_from_hydra(path: Path | None) -> int:
    cfg = read_hydra_config(path) or {}
    value = cfg.get("lista_iterations")
    return int(value) if value is not None else 5


def read_polar_settings_from_hydra(path: Path | None) -> dict[str, object]:
    cfg = read_hydra_config(path) or {}
    return {
        "k_max": cfg.get("k_max"),
        "lista_iterations": cfg.get("lista_iterations"),
        "presence_estimator": cfg.get("presence_estimator"),
        "presence_alpha": cfg.get("presence_alpha"),
        "tau_presence_eval": cfg.get("tau_presence_eval"),
        "gain_feature": cfg.get("gain_feature"),
    }


def read_fully_polar_settings_from_hydra(path: Path | None) -> dict[str, object]:
    cfg = read_hydra_config(path) or {}
    return {
        "k_max": cfg.get("k_max"),
        "lista_iterations": cfg.get("lista_iterations"),
        "presence_estimator": cfg.get("presence_estimator"),
        "sign_estimator": cfg.get("sign_estimator"),
        "presence_alpha": cfg.get("presence_alpha"),
        "tau_presence_eval": cfg.get("tau_presence_eval") or cfg.get("presence_tau_eval"),
    }


def resolve_stress_value(
    cli_value: object | None,
    hydra_values: dict[str, object],
    key: str,
    default: object,
) -> object:
    if cli_value is not None:
        return cli_value
    value = hydra_values.get(key)
    if value is not None:
        return value
    return default


def is_strict_linear_decoder(state_dict: dict[str, torch.Tensor]) -> bool:
    decoder_keys = sorted(k for k in state_dict if k.startswith("decoder."))
    return decoder_keys == ["decoder.net.0.bias", "decoder.net.0.weight"]


def is_old_mlp_decoder(state_dict: dict[str, torch.Tensor]) -> bool:
    decoder_keys = sorted(k for k in state_dict if k.startswith("decoder."))
    return decoder_keys == [
        "decoder.net.0.bias",
        "decoder.net.0.weight",
        "decoder.net.2.bias",
        "decoder.net.2.weight",
        "decoder.net.4.bias",
        "decoder.net.4.weight",
    ]


def is_linear_lista_encoder(state_dict: dict[str, torch.Tensor]) -> bool:
    required = {
        "encoder.input_proj.weight",
        "encoder.recurrent.weight",
        "encoder.head_k.weight",
        "encoder.head_theta.weight",
        "encoder.head_pi.weight",
        "encoder.log_threshold",
    }
    return required.issubset(state_dict.keys())


def is_polar_linear_lista_encoder(state_dict: dict[str, torch.Tensor]) -> bool:
    required = {
        "encoder.shape_input_proj.weight",
        "encoder.shape_recurrent.weight",
        "encoder.head_presence.weight",
        "encoder.head_sign.weight",
        "encoder.gamma_fc1.weight",
        "encoder.gamma_fc2.weight",
        "encoder.head_k.weight",
        "encoder.head_theta.weight",
        "encoder.log_threshold",
    }
    return required.issubset(state_dict.keys())


def is_fully_polar_linear_lista_encoder(state_dict: dict[str, torch.Tensor]) -> bool:
    required = {
        "encoder.shape_input_proj.weight",
        "encoder.shape_recurrent.weight",
        "encoder.head_presence.weight",
        "encoder.head_sign.weight",
        "encoder.gamma_fc1.weight",
        "encoder.gamma_fc2.weight",
        "encoder.head_k.weight",
        "encoder.head_theta.weight",
        "encoder.log_threshold",
    }
    if not required.issubset(state_dict.keys()):
        return False
    gamma_in_dim = state_dict["encoder.gamma_fc1.weight"].shape[1]
    n_atoms = state_dict["encoder.shape_input_proj.weight"].shape[0]
    return gamma_in_dim == n_atoms and state_dict["encoder.head_sign.weight"].shape[0] == n_atoms * 2


def has_lista_structure_proj(state_dict: dict[str, torch.Tensor]) -> bool:
    return {
        "encoder.structure_proj.weight",
        "encoder.structure_proj.bias",
    }.issubset(state_dict.keys())


def soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.relu(x.abs() - threshold)


def normalized_dictionary(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    weight = state_dict["latent.dictionary.weight"].detach().float()
    return F.normalize(weight, p=2, dim=0)


def decoder_forward_old_nonlinear(state_dict: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    w0, b0 = state_dict["decoder.net.0.weight"], state_dict["decoder.net.0.bias"]
    w2, b2 = state_dict["decoder.net.2.weight"], state_dict["decoder.net.2.bias"]
    w4, b4 = state_dict["decoder.net.4.weight"], state_dict["decoder.net.4.bias"]
    h = F.linear(z, w0.float(), b0.float())
    h = F.relu(h)
    h = F.linear(h, w2.float(), b2.float())
    h = F.relu(h)
    return F.linear(h, w4.float(), b4.float())


def effective_atoms_signal_space(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    dict_w = normalized_dictionary(state_dict)
    dec_w = state_dict["decoder.net.0.weight"].detach().float()
    eff = dec_w @ dict_w
    return F.normalize(eff.T, p=2, dim=1)


def nonlinear_atom_responses_signal_space(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    dict_w = normalized_dictionary(state_dict)
    z0 = torch.zeros(1, dict_w.shape[0], dtype=torch.float32)
    baseline = decoder_forward_old_nonlinear(state_dict, z0).squeeze(0)
    atoms = []
    for atom_idx in range(dict_w.shape[1]):
        z = dict_w[:, atom_idx].unsqueeze(0)
        curve = decoder_forward_old_nonlinear(state_dict, z).squeeze(0) - baseline
        atoms.append(curve)
    return F.normalize(torch.stack(atoms, dim=0), p=2, dim=1)


def manual_deterministic_recon_old(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    k_min: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w0, b0 = state_dict["encoder.net.0.weight"], state_dict["encoder.net.0.bias"]
    w2, b2 = state_dict["encoder.net.2.weight"], state_dict["encoder.net.2.bias"]
    w4, b4 = state_dict["encoder.net.4.weight"], state_dict["encoder.net.4.bias"]
    fc_w, fc_b = state_dict["latent.fc_params.weight"], state_dict["latent.fc_params.bias"]
    dict_w = normalized_dictionary(state_dict)
    x_flat = x.view(x.size(0), -1).float()
    h = F.linear(x_flat, w0.float(), b0.float())
    h = F.relu(h)
    h = F.linear(h, w2.float(), b2.float())
    h = F.relu(h)
    h = F.linear(h, w4.float(), b4.float())

    params = F.linear(h, fc_w.float(), fc_b.float())
    n_atoms = dict_w.shape[1]
    raw_k = params[:, :n_atoms]
    raw_theta = params[:, n_atoms : 2 * n_atoms]
    logits = params[:, 2 * n_atoms :].view(x.size(0), n_atoms, 3)

    k = F.softplus(raw_k) + k_min
    theta = F.softplus(raw_theta) + 1e-6
    gamma = k * theta
    idx = logits.argmax(dim=-1)
    delta = F.one_hot(idx, 3).float()[..., 2] - F.one_hot(idx, 3).float()[..., 0]
    b = gamma * delta
    z = F.linear(b, dict_w.float())
    if is_strict_linear_decoder(state_dict):
        dec_w = state_dict["decoder.net.0.weight"].detach().float()
        dec_b = state_dict["decoder.net.0.bias"].detach().float()
        recon = F.linear(z, dec_w, dec_b).view(x.size(0), 1, -1)
    elif is_old_mlp_decoder(state_dict):
        recon = decoder_forward_old_nonlinear(state_dict, z).view(x.size(0), 1, -1)
    else:
        raise RuntimeError("Unsupported old-style decoder for sparse recovery.")
    return recon, {"k": k, "theta": theta, "delta": delta, "gamma": gamma, "B": b}


def manual_deterministic_recon_current(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    k_min: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w0, b0 = state_dict["encoder.net.0.weight"], state_dict["encoder.net.0.bias"]
    w2, b2 = state_dict["encoder.net.2.weight"], state_dict["encoder.net.2.bias"]
    w4, b4 = state_dict["encoder.net.4.weight"], state_dict["encoder.net.4.bias"]
    cp_w, cp_b = state_dict["latent.conv_params.weight"], state_dict["latent.conv_params.bias"]
    dict_w = normalized_dictionary(state_dict)
    dec_w = state_dict["decoder.net.0.weight"].detach().float()
    dec_b = state_dict["decoder.net.0.bias"].detach().float()

    x_flat = x.view(x.size(0), -1).float()
    h = F.linear(x_flat, w0.float(), b0.float())
    h = F.relu(h)
    h = F.linear(h, w2.float(), b2.float())
    h = F.relu(h)
    h = F.linear(h, w4.float(), b4.float())

    params = F.conv1d(h.unsqueeze(-1), cp_w.float(), cp_b.float()).squeeze(-1)
    n_atoms = dict_w.shape[1]
    n_logits = 3 if params.shape[1] == n_atoms * 5 else 2
    raw_k = params[:, :n_atoms]
    raw_theta = params[:, n_atoms : 2 * n_atoms]
    logits = params[:, 2 * n_atoms :].view(x.size(0), n_atoms, n_logits)

    k = F.softplus(raw_k) + k_min
    theta = F.softplus(raw_theta) + 1e-6
    gamma = k * theta
    idx = logits.argmax(dim=-1)
    if n_logits == 3:
        delta = F.one_hot(idx, 3).float()[..., 2] - F.one_hot(idx, 3).float()[..., 0]
    else:
        delta = F.one_hot(idx, 2).float()[..., 1]
    b = gamma * delta
    z = F.linear(b, dict_w.float())
    recon = F.linear(z, dec_w, dec_b).view(x.size(0), 1, -1)
    return recon, {"k": k, "theta": theta, "delta": delta, "gamma": gamma, "B": b}


def manual_deterministic_recon_linear_lista(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    k_min: float,
    lista_iterations: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dict_w = normalized_dictionary(state_dict)
    dec_w = state_dict["decoder.net.0.weight"].detach().float()
    dec_b = state_dict["decoder.net.0.bias"].detach().float()

    x_flat = x.view(x.size(0), -1).float()
    drive = F.linear(
        x_flat,
        state_dict["encoder.input_proj.weight"].float(),
        state_dict["encoder.input_proj.bias"].float(),
    )
    threshold = state_dict["encoder.log_threshold"].float().exp().unsqueeze(0)
    h = soft_threshold(drive, threshold)
    for _ in range(max(lista_iterations - 1, 0)):
        proposal = drive + F.linear(h, state_dict["encoder.recurrent.weight"].float())
        h = soft_threshold(proposal, threshold)

    raw_k = F.linear(h, state_dict["encoder.head_k.weight"].float(), state_dict["encoder.head_k.bias"].float())
    raw_theta = F.linear(
        h,
        state_dict["encoder.head_theta.weight"].float(),
        state_dict["encoder.head_theta.bias"].float(),
    )
    if has_lista_structure_proj(state_dict):
        h_struct = F.normalize(h, p=2, dim=1, eps=1e-6)
        h_struct = torch.tanh(
            F.linear(
                h_struct,
                state_dict["encoder.structure_proj.weight"].float(),
                state_dict["encoder.structure_proj.bias"].float(),
            )
        )
    else:
        h_struct = h
    logits_raw = F.linear(
        h_struct,
        state_dict["encoder.head_pi.weight"].float(),
        state_dict["encoder.head_pi.bias"].float(),
    )
    n_atoms = dict_w.shape[1]
    n_logits = logits_raw.shape[1] // n_atoms

    k = F.softplus(raw_k).unsqueeze(-1) + k_min
    theta = F.softplus(raw_theta).unsqueeze(-1) + 1e-4
    logits = logits_raw.view(x.size(0), n_atoms, n_logits).unsqueeze(2)
    idx = logits.argmax(dim=-1)
    delta_one_hot = F.one_hot(idx, n_logits).float()
    if n_logits == 3:
        delta = delta_one_hot[..., 2] - delta_one_hot[..., 0]
    else:
        delta = delta_one_hot[..., 1]
    gamma = k * theta
    b = gamma * delta
    b_squeezed = b.squeeze(-1)
    delta_squeezed = delta.squeeze(-1)
    gamma_squeezed = gamma.squeeze(-1)
    k_squeezed = k.squeeze(-1)
    theta_squeezed = theta.squeeze(-1)
    logits_squeezed = logits.squeeze(2)
    z = F.linear(b_squeezed, dict_w.float())
    recon = F.linear(z, dec_w, dec_b).view(x.size(0), 1, -1)
    return recon, {
        "k": k_squeezed,
        "theta": theta_squeezed,
        "delta": delta_squeezed,
        "gamma": gamma_squeezed,
        "B": b_squeezed,
        "logits": logits_squeezed,
    }


def manual_deterministic_recon_polar_linear_lista(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    k_min: float,
    k_max: float,
    lista_iterations: int,
    presence_estimator: str,
    presence_alpha: float,
    tau_presence_eval: float,
    gain_feature: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dict_w = normalized_dictionary(state_dict)
    dec_w = state_dict["decoder.net.0.weight"].detach().float()
    dec_b = state_dict["decoder.net.0.bias"].detach().float()

    x_flat = x.view(x.size(0), -1).float()
    input_scale = x_flat.norm(dim=1, keepdim=True)
    x_shape = x_flat / (input_scale + 1e-6)

    drive = F.linear(
        x_shape,
        state_dict["encoder.shape_input_proj.weight"].float(),
        state_dict["encoder.shape_input_proj.bias"].float(),
    )
    threshold = state_dict["encoder.log_threshold"].float().exp().unsqueeze(0)
    h_delta = soft_threshold(drive, threshold)
    for _ in range(max(lista_iterations - 1, 0)):
        proposal = drive + F.linear(h_delta, state_dict["encoder.shape_recurrent.weight"].float())
        h_delta = soft_threshold(proposal, threshold)

    if gain_feature == "log_l2":
        gain_scalar = torch.log(input_scale + 1e-6)
    elif gain_feature == "l2":
        gain_scalar = input_scale
    else:
        raise ValueError(f"Unsupported gain_feature: {gain_feature}")

    gamma_in = torch.cat([x_flat, gain_scalar, h_delta], dim=1)
    h_gamma = F.relu(
        F.linear(
            gamma_in,
            state_dict["encoder.gamma_fc1.weight"].float(),
            state_dict["encoder.gamma_fc1.bias"].float(),
        )
    )
    h_gamma = F.relu(
        F.linear(
            h_gamma,
            state_dict["encoder.gamma_fc2.weight"].float(),
            state_dict["encoder.gamma_fc2.bias"].float(),
        )
    )

    raw_k = F.linear(h_gamma, state_dict["encoder.head_k.weight"].float(), state_dict["encoder.head_k.bias"].float())
    raw_theta = F.linear(
        h_gamma,
        state_dict["encoder.head_theta.weight"].float(),
        state_dict["encoder.head_theta.bias"].float(),
    )
    k = F.softplus(raw_k) + k_min
    if k_max < float("inf"):
        k = torch.clamp(k, max=k_max)
    theta = F.softplus(raw_theta) + 1e-4
    k = k.unsqueeze(-1)
    theta = theta.unsqueeze(-1)

    presence_logits = F.linear(
        h_delta,
        state_dict["encoder.head_presence.weight"].float(),
        state_dict["encoder.head_presence.bias"].float(),
    ).view(x.size(0), dict_w.shape[1], 2).unsqueeze(2)
    sign_scores = F.linear(
        h_delta,
        state_dict["encoder.head_sign.weight"].float(),
        state_dict["encoder.head_sign.bias"].float(),
    ).unsqueeze(-1)

    presence_soft, _ = binary_presence_projection(
        presence_logits,
        estimator=presence_estimator,
        sampling="soft",
        temp=1.0,
        tau_presence_eval=tau_presence_eval,
        presence_alpha=presence_alpha,
    )
    presence_det, _ = binary_presence_projection(
        presence_logits,
        estimator=presence_estimator,
        sampling="deterministic",
        temp=1.0,
        tau_presence_eval=tau_presence_eval,
        presence_alpha=presence_alpha,
    )
    delta, sign_values = combine_presence_sign(
        presence_det,
        sign_scores,
        sampling="deterministic",
        tau_presence_eval=tau_presence_eval,
    )
    gamma = k * theta
    b = gamma * delta
    b_squeezed = b.squeeze(-1)
    z = F.linear(b_squeezed, dict_w.float())
    recon = F.linear(z, dec_w, dec_b).view(x.size(0), 1, -1)
    return recon, {
        "k": k.squeeze(-1),
        "theta": theta.squeeze(-1),
        "delta": delta.squeeze(-1),
        "gamma": gamma.squeeze(-1),
        "B": b_squeezed,
        "logits": presence_logits.squeeze(2),
        "presence_probs": presence_soft.squeeze(-1),
        "presence_det": presence_det.squeeze(-1),
        "sign_values": sign_values.squeeze(-1),
        "shape_state": h_delta,
        "gain_state": h_gamma,
        "input_scale": input_scale.squeeze(-1),
    }


def manual_deterministic_recon_fully_polar_linear_lista(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    k_min: float,
    k_max: float,
    lista_iterations: int,
    presence_estimator: str,
    sign_estimator: str,
    presence_alpha: float,
    tau_presence_eval: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dict_w = normalized_dictionary(state_dict)
    dec_w = state_dict["decoder.net.0.weight"].detach().float()
    dec_b = state_dict["decoder.net.0.bias"].detach().float()

    x_flat = x.view(x.size(0), -1).float()
    input_scale = x_flat.norm(dim=1, keepdim=True) + 1e-6
    x_shape = x_flat / input_scale

    drive = F.linear(
        x_shape,
        state_dict["encoder.shape_input_proj.weight"].float(),
        state_dict["encoder.shape_input_proj.bias"].float(),
    )
    threshold = state_dict["encoder.log_threshold"].float().exp().unsqueeze(0)
    h_delta = soft_threshold(drive, threshold)
    for _ in range(max(lista_iterations - 1, 0)):
        proposal = drive + F.linear(h_delta, state_dict["encoder.shape_recurrent.weight"].float())
        h_delta = soft_threshold(proposal, threshold)

    h_gamma = F.relu(
        F.linear(
            h_delta,
            state_dict["encoder.gamma_fc1.weight"].float(),
            state_dict["encoder.gamma_fc1.bias"].float(),
        )
    )
    h_gamma = F.relu(
        F.linear(
            h_gamma,
            state_dict["encoder.gamma_fc2.weight"].float(),
            state_dict["encoder.gamma_fc2.bias"].float(),
        )
    )

    raw_k = F.linear(h_gamma, state_dict["encoder.head_k.weight"].float(), state_dict["encoder.head_k.bias"].float())
    raw_theta = F.linear(
        h_gamma,
        state_dict["encoder.head_theta.weight"].float(),
        state_dict["encoder.head_theta.bias"].float(),
    )
    k = F.softplus(raw_k) + k_min
    if k_max < float("inf"):
        k = torch.clamp(k, max=k_max)
    theta_tilde = F.softplus(raw_theta) + 1e-4
    theta_final = theta_tilde * input_scale
    k = k.unsqueeze(-1)
    theta_tilde = theta_tilde.unsqueeze(-1)
    theta_final = theta_final.unsqueeze(-1)

    n_atoms = dict_w.shape[1]
    presence_logits = F.linear(
        h_delta,
        state_dict["encoder.head_presence.weight"].float(),
        state_dict["encoder.head_presence.bias"].float(),
    ).view(x.size(0), n_atoms, 2).unsqueeze(2)
    sign_logits = F.linear(
        h_delta,
        state_dict["encoder.head_sign.weight"].float(),
        state_dict["encoder.head_sign.bias"].float(),
    ).view(x.size(0), n_atoms, 2).unsqueeze(2)

    presence_soft, presence_det = binary_presence_projection(
        presence_logits,
        estimator=presence_estimator,
        sampling="soft",
        temp=1.0,
        tau_presence_eval=tau_presence_eval,
        presence_alpha=presence_alpha,
    )
    presence_hard, _ = binary_presence_projection(
        presence_logits,
        estimator=presence_estimator,
        sampling="deterministic",
        temp=1.0,
        tau_presence_eval=tau_presence_eval,
        presence_alpha=presence_alpha,
    )
    sign_soft, sign_det, sign_probs = binary_sign_projection(
        sign_logits,
        estimator=sign_estimator,
        sampling="soft",
        temp=1.0,
    )
    sign_hard, _, _ = binary_sign_projection(
        sign_logits,
        estimator=sign_estimator,
        sampling="deterministic",
        temp=1.0,
    )

    delta = combine_presence_and_sign(presence_hard, sign_hard)
    gamma = k * theta_final
    b = gamma * delta
    b_squeezed = b.squeeze(-1)
    z = F.linear(b_squeezed, dict_w.float())
    recon = F.linear(z, dec_w, dec_b).view(x.size(0), 1, -1)
    return recon, {
        "k": k.squeeze(-1),
        "theta": theta_final.squeeze(-1),
        "theta_tilde": theta_tilde.squeeze(-1),
        "theta_final": theta_final.squeeze(-1),
        "delta": delta.squeeze(-1),
        "gamma": gamma.squeeze(-1),
        "B": b_squeezed,
        "presence_logits": presence_logits.squeeze(2),
        "sign_logits": sign_logits.squeeze(2),
        "presence_probs": presence_soft.squeeze(-1),
        "presence_det": presence_hard.squeeze(-1),
        "sign_values": sign_soft.squeeze(-1),
        "sign_det": sign_hard.squeeze(-1),
        "sign_probs": sign_probs.squeeze(-1),
        "shape_state": h_delta,
        "gain_state": h_gamma,
        "input_scale": input_scale.squeeze(-1),
    }


def support_jaccard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = (a & b).float().sum(dim=1)
    union = (a | b).float().sum(dim=1)
    return torch.where(union > 0, inter / union, torch.ones_like(inter))


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=1, eps=1e-8)


def compute_subspace_scores(atoms: torch.Tensor, max_frequency: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    subspaces, subspace_freqs = build_fourier_subspaces(atoms.shape[1], max_frequency=max_frequency)
    proj = torch.einsum("at,ftd->afd", atoms.float(), subspaces.float())
    scores = proj.norm(dim=-1).clamp(0.0, 1.0)
    best_score, best_idx = scores.max(dim=1)
    return scores, best_score, subspace_freqs


def evaluate_sparse_recovery(
    checkpoint: Path,
    k_min: float,
    spec: SinusoidRecoverySpec,
    min_atom_corr: float,
    min_subspace_score: float,
    hydra_config: Path | None = None,
) -> dict[str, object]:
    batch = generate_sinusoid_recovery_batch(spec)
    x = batch["x"]
    gt_support = batch["gt_support"]
    gt_amp_scores = batch["gt_amp_scores"]

    state_dict = load_state_dict(checkpoint)
    polar_settings = read_polar_settings_from_hydra(hydra_config)
    fully_polar_settings = read_fully_polar_settings_from_hydra(hydra_config)
    if is_fully_polar_linear_lista_encoder(state_dict):
        recon, info = manual_deterministic_recon_fully_polar_linear_lista(
            state_dict,
            x,
            k_min=k_min,
            k_max=float(fully_polar_settings.get("k_max") or float("inf")),
            lista_iterations=int(fully_polar_settings.get("lista_iterations") or 5),
            presence_estimator=str(fully_polar_settings.get("presence_estimator") or "gumbel_binary"),
            sign_estimator=str(fully_polar_settings.get("sign_estimator") or "gumbel_binary"),
            presence_alpha=float(fully_polar_settings.get("presence_alpha") or 1.5),
            tau_presence_eval=float(fully_polar_settings.get("tau_presence_eval") or 0.5),
        )
    elif is_polar_linear_lista_encoder(state_dict):
        recon, info = manual_deterministic_recon_polar_linear_lista(
            state_dict,
            x,
            k_min=k_min,
            k_max=float(polar_settings.get("k_max") or float("inf")),
            lista_iterations=int(polar_settings.get("lista_iterations") or 5),
            presence_estimator=str(polar_settings.get("presence_estimator") or "gumbel_binary"),
            presence_alpha=float(polar_settings.get("presence_alpha") or 1.5),
            tau_presence_eval=float(polar_settings.get("tau_presence_eval") or 0.5),
            gain_feature=str(polar_settings.get("gain_feature") or "log_l2"),
        )
    elif is_linear_lista_encoder(state_dict):
        recon, info = manual_deterministic_recon_linear_lista(
            state_dict,
            x,
            k_min=k_min,
            lista_iterations=read_lista_iterations_from_hydra(hydra_config),
        )
    elif "latent.fc_params.weight" in state_dict:
        recon, info = manual_deterministic_recon_old(state_dict, x, k_min=k_min)
    elif "latent.conv_params.weight" in state_dict:
        recon, info = manual_deterministic_recon_current(state_dict, x, k_min=k_min)
    else:
        raise RuntimeError("Unsupported checkpoint: cannot infer latent parameter head.")

    if is_strict_linear_decoder(state_dict):
        atoms = effective_atoms_signal_space(state_dict)
    elif is_old_mlp_decoder(state_dict):
        atoms = nonlinear_atom_responses_signal_space(state_dict)
    else:
        raise RuntimeError("Sparse recovery currently supports only strict-linear or old MLP decoders.")

    bank, labels = build_fourier_bank(atoms.shape[1], max_frequency=spec.max_frequency)
    corr = torch.abs(atoms.float() @ bank.float().T)
    best_corr, best_idx = corr.max(dim=1)
    best_labels = [labels[int(idx)] for idx in best_idx]

    b = info["B"].detach().float()
    active = (info["delta"].detach() != 0)
    pred_scores = torch.zeros(spec.n_samples, spec.max_frequency, dtype=torch.float32)
    for atom_idx in range(b.shape[1]):
        _, freq_bin = best_labels[atom_idx]
        if float(best_corr[atom_idx].item()) < min_atom_corr:
            continue
        freq_idx = freq_bin - 1
        pred_scores[:, freq_idx] += active[:, atom_idx].float() * b[:, atom_idx].abs() * float(best_corr[atom_idx].item())

    support_metrics = support_metrics_from_scores(pred_scores, gt_support, gt_amp_scores)

    atom_subspace_scores, best_subspace_score, subspace_freqs = compute_subspace_scores(
        atoms, max_frequency=spec.max_frequency
    )
    pred_scores_subspace = torch.zeros(spec.n_samples, len(subspace_freqs), dtype=torch.float32)
    for atom_idx in range(b.shape[1]):
        valid = atom_subspace_scores[atom_idx] >= min_subspace_score
        if not bool(valid.any()):
            continue
        contrib = active[:, atom_idx].float() * b[:, atom_idx].abs()
        pred_scores_subspace += contrib.unsqueeze(1) * (
            atom_subspace_scores[atom_idx] * valid.float()
        ).unsqueeze(0)

    if pred_scores_subspace.shape[1] == gt_support.shape[1]:
        gt_support_subspace = gt_support
        gt_amp_subspace = gt_amp_scores
    else:
        gt_support_subspace = gt_support[:, : pred_scores_subspace.shape[1]]
        gt_amp_subspace = gt_amp_scores[:, : pred_scores_subspace.shape[1]]
    subspace_metrics = support_metrics_from_scores(
        pred_scores_subspace,
        gt_support_subspace,
        gt_amp_subspace,
    )

    n_active_frame = float(active.float().sum(dim=1).mean().item())
    n_active_total = n_active_frame
    recon_mse_per_example = float(F.mse_loss(recon, x, reduction="sum").item() / x.size(0))
    recon_mse_mean = float(F.mse_loss(recon, x, reduction="mean").item())

    invariance_eval: dict[str, float] = {}
    if is_fully_polar_linear_lista_encoder(state_dict):
        base_support = active.bool()
        base_gamma = info["gamma"].detach().float()
        base_theta = info["theta_final"].detach().float()
        base_shape = info["shape_state"].detach().float()
        base_presence = info["presence_probs"].detach().float()
        base_sign = info["sign_det"].detach().float()
        support_consistency_vals = []
        gamma_equiv_vals = []
        theta_equiv_vals = []
        shape_cos_vals = []
        sign_stability_vals = []
        for gain in (0.5, 2.0, 10.0):
            _, scaled_info = manual_deterministic_recon_fully_polar_linear_lista(
                state_dict,
                x * gain,
                k_min=k_min,
                k_max=float(fully_polar_settings.get("k_max") or float("inf")),
                lista_iterations=int(fully_polar_settings.get("lista_iterations") or 5),
                presence_estimator=str(fully_polar_settings.get("presence_estimator") or "gumbel_binary"),
                sign_estimator=str(fully_polar_settings.get("sign_estimator") or "gumbel_binary"),
                presence_alpha=float(fully_polar_settings.get("presence_alpha") or 1.5),
                tau_presence_eval=float(fully_polar_settings.get("tau_presence_eval") or 0.5),
            )
            scaled_support = (scaled_info["delta"].detach() != 0).bool()
            scaled_gamma = scaled_info["gamma"].detach().float()
            scaled_theta = scaled_info["theta_final"].detach().float()
            scaled_shape = scaled_info["shape_state"].detach().float()
            scaled_sign = scaled_info["sign_det"].detach().float()
            support_consistency_vals.append(float(support_jaccard(base_support, scaled_support).mean().item()))
            gamma_equiv_vals.append(
                float(
                    ((scaled_gamma - gain * base_gamma).abs() / (gain * base_gamma.abs() + 1e-6))
                    .mean()
                    .item()
                )
            )
            theta_equiv_vals.append(
                float(
                    ((scaled_theta - gain * base_theta).abs() / (gain * base_theta.abs() + 1e-6))
                    .mean()
                    .item()
                )
            )
            shape_cos_vals.append(float(cosine_rows(base_shape, scaled_shape).mean().item()))
            sign_union = base_support | scaled_support
            if bool(sign_union.any()):
                sign_match = ((base_sign == scaled_sign) | (~sign_union)).float()
                sign_stability_vals.append(float(sign_match.mean().item()))
            else:
                sign_stability_vals.append(1.0)

        presence_clamped = base_presence.clamp(1e-8, 1.0 - 1e-8)
        sign_probs = info["sign_probs"].detach().float().clamp(1e-8, 1.0 - 1e-8)
        invariance_eval = {
            "support_consistency_scaled_mean": float(sum(support_consistency_vals) / len(support_consistency_vals)),
            "gamma_equivariance_error_mean": float(sum(gamma_equiv_vals) / len(gamma_equiv_vals)),
            "theta_equivariance_error_mean": float(sum(theta_equiv_vals) / len(theta_equiv_vals)),
            "presence_entropy_mean": float(
                (-(presence_clamped * presence_clamped.log() + (1.0 - presence_clamped) * (1.0 - presence_clamped).log()))
                .mean()
                .item()
            ),
            "presence_exact_zero_fraction": float((base_presence == 0).float().mean().item()),
            "shape_invariance_cosine_mean": float(sum(shape_cos_vals) / len(shape_cos_vals)),
            "presence_density_mean": float(base_presence.mean().item()),
            "sign_entropy_mean": float(
                (-(sign_probs * sign_probs.log() + (1.0 - sign_probs) * (1.0 - sign_probs).log()))
                .mean()
                .item()
            ),
            "sign_stability_scaled_mean": float(sum(sign_stability_vals) / len(sign_stability_vals)),
        }
    elif is_polar_linear_lista_encoder(state_dict):
        base_support = active.bool()
        base_gamma = info["gamma"].detach().float()
        base_shape = info["shape_state"].detach().float()
        base_presence = info["presence_probs"].detach().float()
        base_sign = torch.sign(info["sign_values"].detach().float())
        support_consistency_vals = []
        gamma_equiv_vals = []
        shape_cos_vals = []
        sign_stability_vals = []
        for gain in (0.5, 2.0, 10.0):
            _, scaled_info = manual_deterministic_recon_polar_linear_lista(
                state_dict,
                x * gain,
                k_min=k_min,
                k_max=float(polar_settings.get("k_max") or float("inf")),
                lista_iterations=int(polar_settings.get("lista_iterations") or 5),
                presence_estimator=str(polar_settings.get("presence_estimator") or "gumbel_binary"),
                presence_alpha=float(polar_settings.get("presence_alpha") or 1.5),
                tau_presence_eval=float(polar_settings.get("tau_presence_eval") or 0.5),
                gain_feature=str(polar_settings.get("gain_feature") or "log_l2"),
            )
            scaled_support = (scaled_info["delta"].detach() != 0).bool()
            scaled_gamma = scaled_info["gamma"].detach().float()
            scaled_shape = scaled_info["shape_state"].detach().float()
            scaled_sign = torch.sign(scaled_info["sign_values"].detach().float())
            support_consistency_vals.append(float(support_jaccard(base_support, scaled_support).mean().item()))
            gamma_equiv_vals.append(
                float(
                    ((scaled_gamma - gain * base_gamma).abs() / (gain * base_gamma.abs() + 1e-6))
                    .mean()
                    .item()
                )
            )
            shape_cos_vals.append(float(cosine_rows(base_shape, scaled_shape).mean().item()))
            sign_union = base_support | scaled_support
            if bool(sign_union.any()):
                sign_match = ((base_sign == scaled_sign) | (~sign_union)).float()
                sign_stability_vals.append(float(sign_match.mean().item()))
            else:
                sign_stability_vals.append(1.0)

        presence_clamped = base_presence.clamp(1e-8, 1.0 - 1e-8)
        invariance_eval = {
            "support_consistency_scaled_mean": float(sum(support_consistency_vals) / len(support_consistency_vals)),
            "gamma_equivariance_error_mean": float(sum(gamma_equiv_vals) / len(gamma_equiv_vals)),
            "theta_equivariance_error_mean": None,
            "presence_entropy_mean": float(
                (-(presence_clamped * presence_clamped.log() + (1.0 - presence_clamped) * (1.0 - presence_clamped).log()))
                .mean()
                .item()
            ),
            "presence_exact_zero_fraction": float((base_presence == 0).float().mean().item()),
            "shape_invariance_cosine_mean": float(sum(shape_cos_vals) / len(shape_cos_vals)),
            "presence_density_mean": float(base_presence.mean().item()),
            "sign_entropy_mean": None,
            "sign_stability_scaled_mean": float(sum(sign_stability_vals) / len(sign_stability_vals)),
        }

    result = {
        "checkpoint": str(checkpoint),
        "dataset": "sinusoid",
        "dataset_spec": batch["spec"],
        "k_min": k_min,
        "reconstruction": {
            "recon_mse_per_example": recon_mse_per_example,
            "recon_mse_mean": recon_mse_mean,
        },
        "latents": {
            "n_active_frame_mean": n_active_frame,
            "n_active_total_mean": n_active_total,
            "k_mean": float(info["k"].detach().mean().item()),
            "k_active_mean": float(info["k"].detach()[active].mean().item()) if active.any() else 0.0,
            "sparsity": float((~active).float().mean().item()),
            "collapsed": collapse_flag(n_active_total),
        },
        "atoms": {
            "n_atoms": int(atoms.shape[0]),
            "top1_fourier_corr_mean": float(best_corr.mean().item()),
            "top1_fourier_corr_median": float(best_corr.median().item()),
            "high_conf_atom_fraction": float((best_corr >= min_atom_corr).float().mean().item()),
            "best_subspace_score_mean": float(best_subspace_score.mean().item()),
            "best_subspace_score_median": float(best_subspace_score.median().item()),
        },
        "support_eval": {
            "min_atom_corr": min_atom_corr,
            **support_metrics,
        },
        "subspace_eval": {
            "min_subspace_score": min_subspace_score,
            **subspace_metrics,
        },
    }
    if invariance_eval:
        result["invariance_eval"] = invariance_eval
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate sparse recovery on toy sinusoid checkpoints")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hydra-config", type=Path, default=None)
    parser.add_argument("--k-min", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frequency", type=int, default=19)
    parser.add_argument("--min-atom-corr", type=float, default=0.70)
    parser.add_argument("--min-subspace-score", type=float, default=0.70)
    parser.add_argument("--gain-distribution", type=str, default=None)
    parser.add_argument("--gain-min", type=float, default=None)
    parser.add_argument("--gain-max", type=float, default=None)
    parser.add_argument("--normalize-divisor", type=float, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    k_min = args.k_min
    if k_min is None:
        k_min = read_k_min_from_hydra(args.hydra_config)
    if k_min is None:
        k_min = 0.1
    hydra_stress = read_sinusoid_stress_from_hydra(args.hydra_config)

    spec = SinusoidRecoverySpec(
        n_samples=args.n_samples,
        length=args.length,
        n_components=args.n_components,
        seed=args.seed,
        max_frequency=args.max_frequency,
        gain_distribution=str(resolve_stress_value(args.gain_distribution, hydra_stress, "gain_distribution", "none")),
        gain_min=float(resolve_stress_value(args.gain_min, hydra_stress, "gain_min", 1.0)),
        gain_max=float(resolve_stress_value(args.gain_max, hydra_stress, "gain_max", 1.0)),
        normalize_divisor=float(
            resolve_stress_value(args.normalize_divisor, hydra_stress, "normalize_divisor", 4.0)
        ),
    )
    result = evaluate_sparse_recovery(
        checkpoint=args.checkpoint,
        k_min=float(k_min),
        spec=spec,
        min_atom_corr=args.min_atom_corr,
        min_subspace_score=args.min_subspace_score,
        hydra_config=args.hydra_config,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
