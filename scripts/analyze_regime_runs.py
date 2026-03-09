#!/usr/bin/env python
"""Generate paper-ready regime-study tables from stored artifacts and logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.regime_analysis import (
    find_wandb_record,
    load_wandb_run_records,
    maybe_read_json,
    maybe_read_yaml,
    parse_librimix_experiment_log,
    parse_old_train_log,
    write_csv,
    write_markdown_table,
)


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix == ".json":
        return maybe_read_json(path) or {}
    return maybe_read_yaml(path) or {}


def scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def run_registry() -> list[dict[str, Any]]:
    return [
        {
            "run_name": "librimix_direct_mask",
            "dataset": "librimix",
            "objective": "supervised_direct_mask",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_direct_mask_baseline_e120",
            "config_path": REPO_ROOT / "checkpoints/librimix_direct_mask_baseline_e120/config.json",
            "summary_path": REPO_ROOT / "checkpoints/librimix_direct_mask_baseline_e120/summary.json",
            "log_path": REPO_ROOT / "librimix_direct_mask_4722107.err",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_direct_mask_baseline.sh",
        },
        {
            "run_name": "librimix_hybrid_partition",
            "dataset": "librimix",
            "objective": "supervised_hybrid_partition",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_hybrid_partition_n512_w16_s4_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_hybrid_partition_n512_w16_s4_e200/config.json",
            "log_path": REPO_ROOT / "librimix_hybrid_partition_4722099.err",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_hybrid_partition.sh",
        },
        {
            "run_name": "librimix_hybrid_partition_randomcrop",
            "dataset": "librimix",
            "objective": "supervised_hybrid_partition_randomcrop",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_hybrid_partition_randomcrop_n512_w16_s4_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_hybrid_partition_randomcrop_n512_w16_s4_e200/config.json",
            "log_path": REPO_ROOT / "librimix_hybrid_partition_rcrop_4722101.err",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_hybrid_partition_randomcrop.sh",
        },
        {
            "run_name": "librimix_hybrid_partition_klsafe",
            "dataset": "librimix",
            "objective": "supervised_hybrid_partition_klsafe",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_hybrid_partition_klsafe_n512_w16_s4_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_hybrid_partition_klsafe_n512_w16_s4_e200/config.json",
            "summary_path": REPO_ROOT / "checkpoints/librimix_hybrid_partition_klsafe_n512_w16_s4_e200/summary.json",
            "log_path": REPO_ROOT / "librimix_hybrid_partition_klsafe_4722199.err",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_hybrid_partition_klsafe.sh",
        },
        {
            "run_name": "librimix_v5_binary_longcrop",
            "dataset": "librimix",
            "objective": "unsupervised_binary_longcrop",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_v5_binary_longcrop_n512_w16_s4_f512_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_v5_binary_longcrop_n512_w16_s4_f512_e200/.hydra/config.yaml",
            "train_log_path": REPO_ROOT / "librimix_v5_longcrop_4722102.out",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_v5_binary_longcrop.sh",
        },
        {
            "run_name": "librimix_v1",
            "dataset": "librimix",
            "objective": "unsupervised_clustered_bss",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_klt1_pure_200",
            "eval_json_path": REPO_ROOT / "checkpoints/librimix_klt1_pure_200/librimix_eval_metrics.json",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_klt1_pure_200.sh",
        },
        {
            "run_name": "librimix_v2",
            "dataset": "librimix",
            "objective": "unsupervised_clustered_bss",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_v2_klt1_n512_w16_s4_e200",
            "eval_json_path": REPO_ROOT / "checkpoints/librimix_v2_klt1_n512_w16_s4_e200/librimix_eval_metrics.json",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_v2_neurips.sh",
        },
        {
            "run_name": "librimix_v3",
            "dataset": "librimix",
            "objective": "unsupervised_clustered_bss_archsafe",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_v3_archsafe_n512_w16_s4_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_v3_archsafe_n512_w16_s4_e200/.hydra/config.yaml",
            "train_log_path": REPO_ROOT / "checkpoints/librimix_v3_archsafe_n512_w16_s4_e200/train.log",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_v3_archsafe.sh",
        },
        {
            "run_name": "librimix_v4_binary_baseline",
            "dataset": "librimix",
            "objective": "unsupervised_clustered_bss_binary",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_v4_binary_baseline_n512_w16_s4_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_v4_binary_baseline_n512_w16_s4_e200/.hydra/config.yaml",
            "train_log_path": REPO_ROOT / "checkpoints/librimix_v4_binary_baseline_n512_w16_s4_e200/train.log",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_v4_binary_baseline.sh",
        },
        {
            "run_name": "librimix_v4_binary_n256",
            "dataset": "librimix",
            "objective": "unsupervised_clustered_bss_binary",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_v4_binary_n256_w16_s4_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_v4_binary_n256_w16_s4_e200/.hydra/config.yaml",
            "train_log_path": REPO_ROOT / "checkpoints/librimix_v4_binary_n256_w16_s4_e200/train.log",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_v4_binary_n256.sh",
        },
        {
            "run_name": "librimix_v4_binary_n512_s8",
            "dataset": "librimix",
            "objective": "unsupervised_clustered_bss_binary",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_v4_binary_n512_w16_s8_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_v4_binary_n512_w16_s8_e200/.hydra/config.yaml",
            "train_log_path": REPO_ROOT / "checkpoints/librimix_v4_binary_n512_w16_s8_e200/train.log",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_v4_binary_s8.sh",
        },
        {
            "run_name": "librimix_v4_binary_n256_s8",
            "dataset": "librimix",
            "objective": "unsupervised_clustered_bss_binary",
            "checkpoint_dir": REPO_ROOT / "checkpoints/librimix_v4_binary_n256_w16_s8_e200",
            "config_path": REPO_ROOT / "checkpoints/librimix_v4_binary_n256_w16_s8_e200/.hydra/config.yaml",
            "train_log_path": REPO_ROOT / "checkpoints/librimix_v4_binary_n256_w16_s8_e200/train.log",
            "launcher": REPO_ROOT / "scripts/slurm/run_librimix_v4_binary_n256_s8.sh",
        },
    ]


def summarize_run(entry: dict[str, Any], wandb_records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(entry.get("config_path"))
    summary = maybe_read_json(entry.get("summary_path"))
    eval_json = maybe_read_json(entry.get("eval_json_path"))
    new_log = parse_librimix_experiment_log(entry["log_path"]) if entry.get("log_path") else {}
    old_log = parse_old_train_log(entry["train_log_path"]) if entry.get("train_log_path") else {}
    wandb_record = find_wandb_record(
        wandb_records,
        save_dir=str(entry["checkpoint_dir"].relative_to(REPO_ROOT)) if entry.get("checkpoint_dir") else None,
    )
    wandb_summary = wandb_record.get("summary", {}) if wandb_record else {}
    wandb_config = wandb_record.get("config", {}) if wandb_record else {}

    merged_config = dict(wandb_config)
    merged_config.update(config)

    row = {
        "run_name": entry["run_name"],
        "dataset": entry["dataset"],
        "objective": entry["objective"],
        "seed": scalar(merged_config.get("seed")),
        "checkpoint_dir": str(entry.get("checkpoint_dir", "")),
        "launcher": str(entry.get("launcher", "")),
        "n_atoms": scalar(merged_config.get("n_atoms")),
        "latent_dim": scalar(merged_config.get("latent_dim")),
        "motif_width": scalar(merged_config.get("motif_width")),
        "decoder_stride": scalar(merged_config.get("decoder_stride")),
        "structure_mode": scalar(merged_config.get("structure_mode")),
        "match_encoder_decoder_stride": scalar(merged_config.get("match_encoder_decoder_stride")),
        "kl_normalization": scalar(merged_config.get("kl_normalization"), "batch"),
        "delta_prior": scalar(merged_config.get("delta_prior")),
        "temp_anneal_epochs": scalar(merged_config.get("temp_anneal_epochs")),
        "phase1_end": scalar(merged_config.get("phase1_end")),
        "phase2_end": scalar(merged_config.get("phase2_end")),
        "phase3_end": scalar(merged_config.get("phase3_end")),
        "best_epoch": None,
        "best_val_metric": None,
        "last_epoch": None,
        "last_val_metric": None,
        "collapse_epoch": None,
        "oracle_gap_db": None,
        "n_active_frame_best": None,
        "n_active_total_best": None,
        "n_active_frame_final": None,
        "n_active_total_final": None,
        "final_weighted_kl_gamma": None,
        "final_weighted_kl_delta": None,
        "eval_split": None,
    }

    if eval_json:
        metrics = eval_json.get("libri2mix", {}).get("metrics", {})
        train_metrics = eval_json.get("train_metrics_final", {})
        row.update(
            {
                "eval_split": eval_json.get("libri2mix", {}).get("split"),
                "best_val_metric": metrics.get("si_sdri_mean_db"),
                "last_val_metric": metrics.get("si_sdri_mean_db"),
                "best_epoch": train_metrics.get("epoch"),
                "last_epoch": train_metrics.get("epoch"),
                "n_active_frame_final": train_metrics.get("n_active_frame"),
                "n_active_total_final": train_metrics.get("n_active_total"),
            }
        )

    if new_log:
        best_val = new_log.get("best_val") or {}
        last_val = new_log.get("last_val") or {}
        best_epoch = new_log.get("best_epoch")
        last_epoch = new_log.get("last_val_epoch")
        row.update(
            {
                "best_epoch": best_epoch if best_epoch is not None else row["best_epoch"],
                "best_val_metric": best_val.get("si_sdri_db", row["best_val_metric"]),
                "last_epoch": last_epoch if last_epoch is not None else row["last_epoch"],
                "last_val_metric": last_val.get("si_sdri_db", row["last_val_metric"]),
                "collapse_epoch": new_log.get("collapse_epoch"),
                "oracle_gap_db": last_val.get("oracle_gap_db", row["oracle_gap_db"]),
            }
        )
        if best_epoch in new_log.get("train_epochs", {}):
            train_best = new_log["train_epochs"][best_epoch]
            row["n_active_frame_best"] = train_best.get("n_active_frame")
            row["n_active_total_best"] = train_best.get("n_active_total")
        if last_epoch in new_log.get("train_epochs", {}):
            train_last = new_log["train_epochs"][last_epoch]
            row["n_active_frame_final"] = train_last.get("n_active_frame", row["n_active_frame_final"])
            row["n_active_total_final"] = train_last.get("n_active_total", row["n_active_total_final"])
            row["final_weighted_kl_gamma"] = train_last.get("weighted_kl_gamma")
            row["final_weighted_kl_delta"] = train_last.get("weighted_kl_delta")

    if old_log:
        last_metrics = old_log.get("last_metrics") or {}
        row.update(
            {
                "last_epoch": old_log.get("last_epoch", row["last_epoch"]),
                "n_active_frame_final": last_metrics.get("n_active_frame", row["n_active_frame_final"]),
                "n_active_total_final": last_metrics.get("n_active_total", row["n_active_total_final"]),
                "final_weighted_kl_gamma": last_metrics.get("weighted_kl_gamma", row["final_weighted_kl_gamma"]),
                "final_weighted_kl_delta": last_metrics.get("weighted_kl_delta", row["final_weighted_kl_delta"]),
            }
        )

    if summary:
        row["best_epoch"] = summary.get("best_epoch", row["best_epoch"])
        row["best_val_metric"] = summary.get("best_si_sdri_mean_db", row["best_val_metric"])

    if wandb_summary:
        row["oracle_gap_db"] = wandb_summary.get("monitor_bss/oracle_gap_db", row["oracle_gap_db"])
        row["best_val_metric"] = wandb_summary.get("monitor_bss/si_sdri_mean_db", row["best_val_metric"])
        row["last_val_metric"] = wandb_summary.get("monitor_bss/si_sdri_mean_db", row["last_val_metric"])
        row["n_active_frame_final"] = wandb_summary.get("n_active_frame", row["n_active_frame_final"])
        row["n_active_total_final"] = wandb_summary.get("n_active", row["n_active_total_final"])
        if row["best_val_metric"] is not None and row["best_epoch"] is None:
            row["best_epoch"] = wandb_summary.get("epoch")
            row["last_epoch"] = wandb_summary.get("epoch")

    artifact = {
        "metadata": entry,
        "config": merged_config,
        "summary_json": summary,
        "eval_json": eval_json,
        "new_log": new_log,
        "old_log": old_log,
        "wandb_summary": wandb_summary,
        "row": row,
    }
    return row, artifact


def main() -> None:
    wandb_records = load_wandb_run_records(REPO_ROOT)
    summaries_dir = REPO_ROOT / "report/tables/run_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for entry in run_registry():
        row, artifact = summarize_run(entry, wandb_records)
        rows.append(row)
        artifacts[entry["run_name"]] = artifact
        with (summaries_dir / f"{entry['run_name']}.json").open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True, default=str)

    regime_rows = rows
    librimix_rows = [
        {
            "run_name": row["run_name"],
            "objective": row["objective"],
            "structure_mode": row["structure_mode"],
            "n_atoms": row["n_atoms"],
            "motif_width": row["motif_width"],
            "decoder_stride": row["decoder_stride"],
            "match_encoder_decoder_stride": row["match_encoder_decoder_stride"],
            "best_si_sdri_db": row["best_val_metric"],
            "last_si_sdri_db": row["last_val_metric"],
            "oracle_gap_db": row["oracle_gap_db"],
            "collapse_epoch": row["collapse_epoch"],
            "n_active_frame_final": row["n_active_frame_final"],
            "n_active_total_final": row["n_active_total_final"],
            "eval_split": row["eval_split"],
        }
        for row in rows
        if row["dataset"] == "librimix"
    ]
    optimization_rows = [
        {
            "run_name": row["run_name"],
            "objective": row["objective"],
            "kl_normalization": row["kl_normalization"],
            "delta_prior": row["delta_prior"],
            "temp_anneal_epochs": row["temp_anneal_epochs"],
            "phase1_end": row["phase1_end"],
            "phase2_end": row["phase2_end"],
            "phase3_end": row["phase3_end"],
            "best_si_sdri_db": row["best_val_metric"],
            "last_si_sdri_db": row["last_val_metric"],
            "collapse_epoch": row["collapse_epoch"],
            "oracle_gap_db": row["oracle_gap_db"],
            "n_active_frame_best": row["n_active_frame_best"],
            "n_active_total_best": row["n_active_total_best"],
            "n_active_frame_final": row["n_active_frame_final"],
            "n_active_total_final": row["n_active_total_final"],
            "final_weighted_kl_gamma": row["final_weighted_kl_gamma"],
            "final_weighted_kl_delta": row["final_weighted_kl_delta"],
        }
        for row in rows
        if row["run_name"]
        in {
            "librimix_direct_mask",
            "librimix_hybrid_partition",
            "librimix_hybrid_partition_randomcrop",
            "librimix_hybrid_partition_klsafe",
            "librimix_v5_binary_longcrop",
        }
    ]

    tables_dir = REPO_ROOT / "report/tables"
    write_csv(tables_dir / "regime_summary.csv", regime_rows)
    write_csv(tables_dir / "librimix_diagnostics.csv", librimix_rows)
    write_csv(tables_dir / "optimization_ablation.csv", optimization_rows)
    write_markdown_table(tables_dir / "regime_summary.md", regime_rows)
    print(f"Wrote {len(regime_rows)} regime rows to {tables_dir}")


if __name__ == "__main__":
    main()
