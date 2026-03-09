"""Parsing helpers for regime-study reporting artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


OLD_TRAIN_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\[(?P<phase>[^\]]+)\]\s+\|\s+"
    r"loss\s+(?P<loss>[-\d.]+)\s+\|\s+recon\s+(?P<recon>[-\d.]+)\s+\|\s+"
    r"kl_γ\s+(?P<kl_gamma>[-\d.]+)\s+\|\s+kl_δ\s+(?P<kl_delta>[-\d.]+)"
    r"(?:\s+\|\s+wkl_γ\s+(?P<weighted_kl_gamma>[-\d.]+)\s+\|\s+wkl_δ\s+(?P<weighted_kl_delta>[-\d.]+))?"
    r"(?:\s+\|\s+coh\s+(?P<coherence>[-\d.]+))?\s+\|\s+"
    r"k̄=(?P<k_mean>[-\d.]+)\s+k_act=(?P<k_active>[-\d.]+)\s+"
    r"n_act_frame=(?P<n_active_frame>[-\d.]+)\s+n_act_total=(?P<n_active_total>[-\d.]+)/(?P<n_atoms>\d+)\s+"
    r"δ₀=(?P<sparsity_pct>[-\d.]+)%\s+"
    r"(?:collapse=(?P<collapse>\d)\s+)?β=(?P<beta>[-\d.]+)\s+τ=(?P<temp>[-\d.]+)\s+Δdict=(?P<dict_drift>[-\d.]+)"
    r"(?:\s+Δeff=(?P<effective_atom_drift>[-\d.]+))?"
)


def parse_number(value: str) -> float:
    return float(value.rstrip("s"))


def maybe_read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_read_yaml(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def unwrap_wandb_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    unwrapped: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            unwrapped[key] = value["value"]
        else:
            unwrapped[key] = value
    return unwrapped


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        ordered = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_markdown_table(rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> str:
    if not rows:
        return ""
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def load_wandb_run_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for config_path in sorted(root.glob("wandb/run-*/files/config.yaml")):
        run_dir = config_path.parents[1]
        summary_path = config_path.with_name("wandb-summary.json")
        raw_config = maybe_read_yaml(config_path)
        summary = maybe_read_json(summary_path) or {}
        config = unwrap_wandb_config(raw_config)
        records.append(
            {
                "run_dir": run_dir,
                "config": config,
                "summary": summary,
                "save_dir": config.get("save_dir"),
                "wandb_run_name": config.get("wandb_run_name"),
            }
        )
    return records


def find_wandb_record(
    records: list[dict[str, Any]],
    *,
    save_dir: str | None = None,
    wandb_run_name: str | None = None,
) -> dict[str, Any] | None:
    for record in records:
        if save_dir is not None and record.get("save_dir") == save_dir:
            return record
        if wandb_run_name is not None and record.get("wandb_run_name") == wandb_run_name:
            return record
    return None


def parse_old_train_log(path: Path) -> dict[str, Any]:
    epochs: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return {"epochs": epochs, "last_epoch": None}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = OLD_TRAIN_RE.search(line)
        if not match:
            continue
        values = match.groupdict()
        epoch = int(values.pop("epoch"))
        parsed: dict[str, Any] = {"epoch": epoch, "phase": values.pop("phase")}
        for key, value in values.items():
            if value is None:
                continue
            if key == "n_atoms" or key == "collapse":
                parsed[key] = int(value)
            elif key == "sparsity_pct":
                parsed["sparsity"] = float(value) / 100.0
            else:
                parsed[key] = float(value)
        epochs[epoch] = parsed
    last_epoch = max(epochs) if epochs else None
    return {"epochs": epochs, "last_epoch": last_epoch, "last_metrics": epochs.get(last_epoch) if last_epoch else None}


def aggregate_numeric(values: list[Any]) -> dict[str, Any]:
    filtered: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isnan(numeric):
            continue
        filtered.append(numeric)
    if not filtered:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(filtered) / len(filtered)
    std = math.sqrt(sum((value - mean) ** 2 for value in filtered) / len(filtered))
    return {
        "count": len(filtered),
        "mean": mean,
        "std": std,
        "min": min(filtered),
        "max": max(filtered),
    }


def robust_positive_aggregate(values: list[Any], outlier_factor: float = 10.0) -> dict[str, Any]:
    filtered: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isnan(numeric):
            continue
        filtered.append(numeric)
    if not filtered:
        return {
            "count": 0,
            "median": None,
            "stable_mean": None,
            "stable_std": None,
            "outlier_count": 0,
        }
    ordered = sorted(filtered)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 == 1 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    if median <= 0.0:
        stable = ordered
    else:
        stable = [value for value in ordered if value <= median * outlier_factor]
    if not stable:
        stable = ordered
    stats = aggregate_numeric(stable)
    return {
        "count": len(filtered),
        "median": median,
        "stable_mean": stats["mean"],
        "stable_std": stats["std"],
        "outlier_count": len(filtered) - len(stable),
    }


SEED_NAME_RE = re.compile(r"(?:^|_)seed_(?P<seed>\d+)$")


def parse_seed_from_name(name: str) -> int | None:
    match = SEED_NAME_RE.search(name)
    if match is None:
        return None
    return int(match.group("seed"))


def summarize_sinusoid_seed_dir(seed_dir: Path) -> dict[str, Any]:
    config = maybe_read_yaml(seed_dir / ".hydra/config.yaml") or {}
    train_log = parse_old_train_log(seed_dir / "train.log")
    sparse_recovery = maybe_read_json(seed_dir / "sparse_recovery.json")
    baseline_metrics = maybe_read_json(seed_dir / "baseline_metrics.json")
    baseline_sparse_recovery = maybe_read_json(seed_dir / "baseline_sparse_recovery.json")
    last_metrics = train_log.get("last_metrics") or {}
    seed = parse_seed_from_name(seed_dir.name)
    row: dict[str, Any] = {
        "condition": seed_dir.parent.name,
        "seed": seed,
        "status": "pending",
        "model_family": "hsvae",
        "baseline_name": None,
        "last_epoch": train_log.get("last_epoch"),
        "epochs_target": config.get("epochs"),
        "kl_normalization": config.get("kl_normalization"),
        "structure_mode": config.get("structure_mode"),
        "delta_factorization": config.get("delta_factorization"),
        "presence_estimator": config.get("presence_estimator"),
        "sign_estimator": config.get("sign_estimator"),
        "presence_prior": config.get("presence_prior"),
        "sign_prior": config.get("sign_prior"),
        "polar_encoder": config.get("polar_encoder"),
        "fully_polar_encoder": config.get("fully_polar_encoder"),
        "k_min": config.get("k_min"),
        "k_max": config.get("k_max"),
        "beta_gamma_final": config.get("beta_gamma_final"),
        "beta_delta_final": config.get("beta_delta_final"),
        "beta_presence_final": config.get("beta_presence_final"),
        "beta_sign_final": config.get("beta_sign_final"),
        "lambda_presence_consistency_final": config.get("lambda_presence_consistency_final"),
        "gamma_kl_target": config.get("gamma_kl_target"),
        "presence_consistency_target": config.get("presence_consistency_target"),
        "phase1_end": config.get("phase1_end"),
        "phase2_end": config.get("phase2_end"),
        "phase3_end": config.get("phase3_end"),
        "temp_init": config.get("temp_init"),
        "temp_min": config.get("temp_min"),
        "temp_anneal_epochs": config.get("temp_anneal_epochs"),
        "dict_init": config.get("dict_init"),
        "dict_lr_mult": config.get("dict_lr_mult"),
        "gain_distribution": config.get("sinusoid_gain_distribution"),
        "gain_min": config.get("sinusoid_gain_min"),
        "gain_max": config.get("sinusoid_gain_max"),
        "normalize_divisor": config.get("sinusoid_normalize_divisor"),
        "dict_drift": last_metrics.get("dict_drift"),
        "effective_atom_drift": last_metrics.get("effective_atom_drift"),
        "recon_mse_per_example": None,
        "support_precision_mean": None,
        "support_recall_mean": None,
        "support_f1_mean": None,
        "topk_f1_mean": None,
        "subspace_f1_mean": None,
        "amplitude_corr_mean": None,
        "top1_fourier_corr_mean": None,
        "high_conf_atom_fraction": None,
        "support_consistency_scaled_mean": None,
        "gamma_equivariance_error_mean": None,
        "presence_entropy_mean": None,
        "presence_exact_zero_fraction": None,
        "sign_entropy_mean": None,
        "shape_invariance_cosine_mean": None,
        "presence_density_mean": None,
        "sign_stability_scaled_mean": None,
        "theta_equivariance_error_mean": None,
        "n_active_frame_mean": None,
        "n_active_total_mean": None,
        "k_active_mean": None,
        "sparsity": None,
        "collapsed": None,
    }

    if sparse_recovery is not None:
        latents = sparse_recovery.get("latents", {})
        support_eval = sparse_recovery.get("support_eval", {})
        subspace_eval = sparse_recovery.get("subspace_eval", {})
        recon = sparse_recovery.get("reconstruction", {})
        atoms = sparse_recovery.get("atoms", {})
        invariance = sparse_recovery.get("invariance_eval", {})
        dataset_spec = sparse_recovery.get("dataset_spec", {})
        baseline_name = sparse_recovery.get("baseline_name")
        row.update(
            {
                "status": "complete",
                "model_family": "baseline" if baseline_name is not None else "hsvae",
                "baseline_name": baseline_name,
                "recon_mse_per_example": recon.get("recon_mse_per_example"),
                "support_precision_mean": support_eval.get("support_precision_mean"),
                "support_recall_mean": support_eval.get("support_recall_mean"),
                "support_f1_mean": support_eval.get("support_f1_mean"),
                "topk_f1_mean": support_eval.get("topk_f1_mean"),
                "subspace_f1_mean": subspace_eval.get("support_f1_mean"),
                "amplitude_corr_mean": support_eval.get("pred_gt_amplitude_corr_mean"),
                "top1_fourier_corr_mean": atoms.get("top1_fourier_corr_mean"),
                "high_conf_atom_fraction": atoms.get("high_conf_atom_fraction"),
                "support_consistency_scaled_mean": invariance.get("support_consistency_scaled_mean"),
                "gamma_equivariance_error_mean": invariance.get("gamma_equivariance_error_mean"),
                "presence_entropy_mean": invariance.get("presence_entropy_mean"),
                "presence_exact_zero_fraction": invariance.get("presence_exact_zero_fraction"),
                "sign_entropy_mean": invariance.get("sign_entropy_mean"),
                "shape_invariance_cosine_mean": invariance.get("shape_invariance_cosine_mean"),
                "presence_density_mean": invariance.get("presence_density_mean"),
                "sign_stability_scaled_mean": invariance.get("sign_stability_scaled_mean"),
                "theta_equivariance_error_mean": invariance.get("theta_equivariance_error_mean"),
                "n_active_frame_mean": latents.get("n_active_frame_mean"),
                "n_active_total_mean": latents.get("n_active_total_mean"),
                "k_active_mean": latents.get("k_active_mean"),
                "sparsity": latents.get("sparsity"),
                "collapsed": latents.get("collapsed"),
                "k_min": sparse_recovery.get("k_min", row["k_min"]),
                "epochs_target": row["epochs_target"] or config.get("epochs"),
                "gain_distribution": dataset_spec.get("gain_distribution", row["gain_distribution"]),
                "gain_min": dataset_spec.get("gain_min", row["gain_min"]),
                "gain_max": dataset_spec.get("gain_max", row["gain_max"]),
                "normalize_divisor": dataset_spec.get("normalize_divisor", row["normalize_divisor"]),
            }
        )
        return row

    if baseline_metrics is not None:
        if baseline_sparse_recovery is not None:
            baseline_key = str(baseline_sparse_recovery.get("selected_baseline", "omp_33"))
            baselines = baseline_sparse_recovery.get("baselines", {})
            selected = baselines.get(baseline_key, {})
            latents = selected.get("latents", {})
            support_eval = selected.get("support_eval", {})
            subspace_eval = selected.get("subspace_eval", {})
            recon = selected.get("reconstruction", {})
            row.update(
                {
                    "status": "complete",
                    "model_family": "baseline",
                    "baseline_name": baseline_key,
                    "recon_mse_per_example": recon.get("recon_mse_per_example"),
                    "support_precision_mean": support_eval.get("support_precision_mean"),
                    "support_recall_mean": support_eval.get("support_recall_mean"),
                    "support_f1_mean": support_eval.get("support_f1_mean"),
                    "topk_f1_mean": support_eval.get("topk_f1_mean"),
                    "subspace_f1_mean": subspace_eval.get("support_f1_mean"),
                    "amplitude_corr_mean": support_eval.get("pred_gt_amplitude_corr_mean"),
                    "n_active_frame_mean": latents.get("n_active_frame_mean"),
                    "n_active_total_mean": latents.get("n_active_total_mean"),
                    "sparsity": latents.get("sparsity"),
                    "collapsed": latents.get("collapsed"),
                }
            )
            return row
        sparse_ae = baseline_metrics.get("results", {}).get("sparse_ae", {})
        row.update(
            {
                "status": "complete",
                "model_family": "baseline",
                "recon_mse_per_example": sparse_ae.get("recon_mse"),
                "sparsity": sparse_ae.get("sparsity"),
                "collapsed": False,
            }
        )
        return row

    if train_log.get("last_epoch") is not None:
        row.update(
            {
                "status": "running",
                "n_active_frame_mean": last_metrics.get("n_active_frame"),
                "n_active_total_mean": last_metrics.get("n_active_total"),
                "k_active_mean": last_metrics.get("k_active"),
                "sparsity": last_metrics.get("sparsity"),
                "collapsed": bool(last_metrics.get("collapse", 0)),
                "dict_drift": last_metrics.get("dict_drift"),
                "effective_atom_drift": last_metrics.get("effective_atom_drift"),
            }
        )
    return row


def summarize_sinusoid_condition_dir(condition_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate_json = maybe_read_json(condition_dir / "aggregate.json") or {}
    seed_dirs = sorted(path for path in condition_dir.glob("seed_*") if path.is_dir())
    seed_rows = [summarize_sinusoid_seed_dir(seed_dir) for seed_dir in seed_dirs]
    completed = [row for row in seed_rows if row["status"] == "complete"]
    expected_seeds = len(aggregate_json.get("seeds", [])) or len(seed_rows)
    has_running = any(row["status"] == "running" for row in seed_rows)
    status = "pending"
    if completed and len(completed) == expected_seeds and expected_seeds > 0:
        status = "complete"
    elif completed or has_running:
        status = "partial"

    def first_value(key: str) -> Any:
        for row in seed_rows:
            value = row.get(key)
            if value is not None:
                return value
        return None

    support_stats = aggregate_numeric([row.get("support_f1_mean") for row in completed])
    support_precision_stats = aggregate_numeric([row.get("support_precision_mean") for row in completed])
    support_recall_stats = aggregate_numeric([row.get("support_recall_mean") for row in completed])
    topk_stats = aggregate_numeric([row.get("topk_f1_mean") for row in completed])
    amplitude_stats = aggregate_numeric([row.get("amplitude_corr_mean") for row in completed])
    subspace_stats = aggregate_numeric([row.get("subspace_f1_mean") for row in completed])
    atom_corr_stats = aggregate_numeric([row.get("top1_fourier_corr_mean") for row in completed])
    high_conf_stats = aggregate_numeric([row.get("high_conf_atom_fraction") for row in completed])
    support_consistency_stats = aggregate_numeric([row.get("support_consistency_scaled_mean") for row in completed])
    gamma_equiv_stats = aggregate_numeric([row.get("gamma_equivariance_error_mean") for row in completed])
    presence_entropy_stats = aggregate_numeric([row.get("presence_entropy_mean") for row in completed])
    presence_zero_stats = aggregate_numeric([row.get("presence_exact_zero_fraction") for row in completed])
    shape_invariance_stats = aggregate_numeric([row.get("shape_invariance_cosine_mean") for row in completed])
    presence_density_stats = aggregate_numeric([row.get("presence_density_mean") for row in completed])
    sign_stability_stats = aggregate_numeric([row.get("sign_stability_scaled_mean") for row in completed])
    recon_stats = aggregate_numeric([row.get("recon_mse_per_example") for row in completed])
    active_frame_stats = aggregate_numeric([row.get("n_active_frame_mean") for row in completed])
    active_total_stats = aggregate_numeric([row.get("n_active_total_mean") for row in completed])
    k_active_stats = aggregate_numeric([row.get("k_active_mean") for row in completed])
    sparsity_stats = aggregate_numeric([row.get("sparsity") for row in completed])
    collapse_stats = aggregate_numeric([1.0 if row.get("collapsed") else 0.0 for row in completed])

    aggregate_row = {
        "condition": condition_dir.name,
        "status": status,
        "completed_seeds": len(completed),
        "observed_seeds": len(seed_rows),
        "expected_seeds": expected_seeds,
        "model_family": first_value("model_family"),
        "baseline_name": first_value("baseline_name"),
        "structure_mode": first_value("structure_mode"),
        "delta_factorization": first_value("delta_factorization"),
        "presence_estimator": first_value("presence_estimator"),
        "sign_estimator": first_value("sign_estimator"),
        "presence_prior": first_value("presence_prior"),
        "sign_prior": first_value("sign_prior"),
        "polar_encoder": first_value("polar_encoder"),
        "fully_polar_encoder": first_value("fully_polar_encoder"),
        "kl_normalization": first_value("kl_normalization"),
        "k_min": first_value("k_min"),
        "k_max": first_value("k_max"),
        "beta_gamma_final": first_value("beta_gamma_final"),
        "beta_delta_final": first_value("beta_delta_final"),
        "beta_presence_final": first_value("beta_presence_final"),
        "beta_sign_final": first_value("beta_sign_final"),
        "lambda_presence_consistency_final": first_value("lambda_presence_consistency_final"),
        "gamma_kl_target": first_value("gamma_kl_target"),
        "presence_consistency_target": first_value("presence_consistency_target"),
        "phase1_end": first_value("phase1_end"),
        "phase2_end": first_value("phase2_end"),
        "phase3_end": first_value("phase3_end"),
        "temp_init": first_value("temp_init"),
        "temp_min": first_value("temp_min"),
        "temp_anneal_epochs": first_value("temp_anneal_epochs"),
        "dict_init": first_value("dict_init"),
        "dict_lr_mult": first_value("dict_lr_mult"),
        "gain_distribution": first_value("gain_distribution"),
        "gain_min": first_value("gain_min"),
        "gain_max": first_value("gain_max"),
        "normalize_divisor": first_value("normalize_divisor"),
        "support_precision_mean": support_precision_stats["mean"],
        "support_precision_std": support_precision_stats["std"],
        "support_recall_mean": support_recall_stats["mean"],
        "support_recall_std": support_recall_stats["std"],
        "support_f1_mean": support_stats["mean"],
        "support_f1_std": support_stats["std"],
        "topk_f1_mean": topk_stats["mean"],
        "topk_f1_std": topk_stats["std"],
        "subspace_f1_mean": subspace_stats["mean"],
        "subspace_f1_std": subspace_stats["std"],
        "amplitude_corr_mean": amplitude_stats["mean"],
        "amplitude_corr_std": amplitude_stats["std"],
        "top1_fourier_corr_mean": atom_corr_stats["mean"],
        "top1_fourier_corr_std": atom_corr_stats["std"],
        "high_conf_atom_fraction": high_conf_stats["mean"],
        "high_conf_atom_fraction_std": high_conf_stats["std"],
        "support_consistency_scaled_mean": support_consistency_stats["mean"],
        "gamma_equivariance_error_mean": gamma_equiv_stats["mean"],
        "presence_entropy_mean": presence_entropy_stats["mean"],
        "presence_exact_zero_fraction": presence_zero_stats["mean"],
        "sign_entropy_mean": aggregate_numeric([row.get("sign_entropy_mean") for row in completed])["mean"],
        "shape_invariance_cosine_mean": shape_invariance_stats["mean"],
        "presence_density_mean": presence_density_stats["mean"],
        "sign_stability_scaled_mean": sign_stability_stats["mean"],
        "theta_equivariance_error_mean": aggregate_numeric([row.get("theta_equivariance_error_mean") for row in completed])["mean"],
        "recon_mse_mean": recon_stats["mean"],
        "recon_mse_std": recon_stats["std"],
        "n_active_frame_mean": active_frame_stats["mean"],
        "n_active_total_mean": active_total_stats["mean"],
        "k_active_mean": k_active_stats["mean"],
        "sparsity_mean": sparsity_stats["mean"],
        "collapse_rate": collapse_stats["mean"],
    }
    return seed_rows, aggregate_row


def summarize_training_run_dir(run_dir: Path, *, dataset: str, run_name: str | None = None, regime: str | None = None) -> dict[str, Any]:
    config = maybe_read_yaml(run_dir / ".hydra/config.yaml") or {}
    train_log = parse_old_train_log(run_dir / "train.log")
    last_metrics = train_log.get("last_metrics") or {}
    last_epoch = train_log.get("last_epoch")
    epochs_target = config.get("epochs")
    status = "pending"
    if last_epoch is not None:
        status = "running"
        if epochs_target is not None and last_epoch >= int(epochs_target):
            status = "complete"
    row = {
        "run_name": run_name or run_dir.name,
        "dataset": dataset,
        "regime": regime,
        "seed": config.get("seed", parse_seed_from_name(run_dir.name)),
        "status": status,
        "last_epoch": last_epoch,
        "epochs_target": epochs_target,
        "progress": (float(last_epoch) / float(epochs_target)) if last_epoch is not None and epochs_target else None,
        "structure_mode": config.get("structure_mode"),
        "kl_normalization": config.get("kl_normalization"),
        "k_min": config.get("k_min"),
        "k_max": config.get("k_max"),
        "masked_recon": config.get("masked_recon"),
        "denoise": config.get("denoise"),
        "gamma_kl_target": config.get("gamma_kl_target"),
        "presence_consistency_target": config.get("presence_consistency_target"),
        "lambda_presence_consistency_final": config.get("lambda_presence_consistency_final"),
        "dict_init": config.get("dict_init"),
        "dict_lr_mult": config.get("dict_lr_mult"),
        "recon": last_metrics.get("recon"),
        "collapse": bool(last_metrics.get("collapse", 0)) if last_metrics else None,
        "n_active_frame": last_metrics.get("n_active_frame"),
        "n_active_total": last_metrics.get("n_active_total"),
        "sparsity": last_metrics.get("sparsity"),
        "k_mean": last_metrics.get("k_mean"),
        "k_active": last_metrics.get("k_active"),
        "weighted_kl_gamma": last_metrics.get("weighted_kl_gamma"),
        "weighted_kl_delta": last_metrics.get("weighted_kl_delta"),
        "dict_drift": last_metrics.get("dict_drift"),
        "effective_atom_drift": last_metrics.get("effective_atom_drift"),
    }
    return row


def _normalize_key(key: str) -> str:
    mapping = {
        "loss": "total_loss",
        "source": "source_loss",
        "mix": "mix_loss",
        "kl_g": "kl_gamma",
        "kl_d": "kl_delta",
        "wkl_g": "weighted_kl_gamma",
        "wkl_d": "weighted_kl_delta",
        "n_act_frame": "n_active_frame",
        "n_act_total": "n_active_total",
        "H(assign)": "assign_entropy",
        "SI-SDRi": "si_sdri_db",
        "SI-SDR": "si_sdr_db",
    }
    return mapping.get(key, key)


def parse_librimix_experiment_log(path: Path) -> dict[str, Any]:
    train_epochs: dict[int, dict[str, Any]] = {}
    val_epochs: dict[int, dict[str, Any]] = {}
    oracle_epochs: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return {
            "train_epochs": train_epochs,
            "val_epochs": val_epochs,
            "oracle_epochs": oracle_epochs,
            "best_epoch": None,
            "collapse_epoch": None,
        }

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "Validation epoch" in stripped:
            parts = [part.strip() for part in stripped[stripped.index("Validation epoch"):].split("|")]
            epoch = int(parts[0].split()[-1])
            is_oracle = parts[1].lower().startswith("oracle ")
            target = oracle_epochs if is_oracle else val_epochs
            metrics = target.setdefault(epoch, {"epoch": epoch})
            if is_oracle:
                val_epochs.setdefault(epoch, {"epoch": epoch})
            for part in parts[1:]:
                if "=" in part and "eval_n" in part:
                    key, value = part.split("=", 1)
                    metrics[key.strip()] = int(value)
                    continue
                tokens = part.split()
                if not tokens:
                    continue
                if tokens[0] == "oracle" and len(tokens) >= 3:
                    key = _normalize_key(" ".join(tokens[:2]))
                    value = tokens[2]
                else:
                    key = _normalize_key(tokens[0])
                    value = tokens[1] if len(tokens) > 1 else ""
                if key == "oracle SI-SDRi":
                    metrics["oracle_si_sdri_db"] = parse_number(value)
                    val_epochs[epoch]["oracle_si_sdri_db"] = metrics["oracle_si_sdri_db"]
                elif key == "oracle gap":
                    metrics["oracle_gap_db"] = parse_number(value)
                    val_epochs[epoch]["oracle_gap_db"] = metrics["oracle_gap_db"]
                elif key in {"si_sdri_db", "si_sdr_db", "source_loss", "mix_loss"}:
                    metrics[key] = parse_number(value)
            continue

        if "Epoch" not in stripped:
            continue
        epoch_segment = stripped[stripped.index("Epoch") :]
        parts = [part.strip() for part in epoch_segment.split("|")]
        if not parts or not parts[0].startswith("Epoch"):
            continue
        epoch = int(parts[0].split()[1])
        metrics = train_epochs.setdefault(epoch, {"epoch": epoch})
        for part in parts[1:]:
            if part.endswith("s") and part[0].isdigit():
                continue
            if part.startswith("n_act_total"):
                _, value = part.split(maxsplit=1)
                active, total = value.split("/")
                metrics["n_active_total"] = float(active)
                metrics["n_atoms"] = int(total)
                continue
            if part.startswith("H(assign)"):
                _, value = part.split(maxsplit=1)
                metrics["assign_entropy"] = float(value)
                continue
            tokens = part.split(maxsplit=1)
            if len(tokens) != 2:
                continue
            key, value = tokens
            key = _normalize_key(key)
            if key in {
                "total_loss",
                "source_loss",
                "mix_loss",
                "kl_gamma",
                "kl_delta",
                "weighted_kl_gamma",
                "weighted_kl_delta",
                "n_active_frame",
                "temp",
            }:
                metrics[key] = parse_number(value)

    best_epoch = None
    if val_epochs:
        best_epoch = max(val_epochs, key=lambda epoch: val_epochs[epoch].get("si_sdri_db", float("-inf")))
    collapse_epoch = None
    for epoch in sorted(train_epochs):
        metrics = train_epochs[epoch]
        if metrics.get("n_active_frame", float("inf")) < 1.0 and metrics.get("n_active_total", float("inf")) < 1.0:
            collapse_epoch = epoch
            break
    last_val_epoch = max(val_epochs) if val_epochs else None
    return {
        "train_epochs": train_epochs,
        "val_epochs": val_epochs,
        "oracle_epochs": oracle_epochs,
        "best_epoch": best_epoch,
        "best_val": val_epochs.get(best_epoch) if best_epoch is not None else None,
        "last_val_epoch": last_val_epoch,
        "last_val": val_epochs.get(last_val_epoch) if last_val_epoch is not None else None,
        "collapse_epoch": collapse_epoch,
    }
