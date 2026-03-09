#!/usr/bin/env python
"""Aggregate live paper-batch experiments into paper-ready tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.regime_analysis import (
    aggregate_numeric,
    robust_positive_aggregate,
    render_markdown_table,
    summarize_sinusoid_condition_dir,
    summarize_training_run_dir,
    write_csv,
)


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def summarize_mnist_root(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.glob("*_seed_*") if path.is_dir()):
        parts = run_dir.name.split("_seed_")
        regime = parts[0] if len(parts) == 2 else None
        row = summarize_training_run_dir(run_dir, dataset="mnist", regime=regime)
        seed_rows.append(row)

    aggregate_rows: list[dict[str, Any]] = []
    regimes = sorted({row["regime"] for row in seed_rows if row.get("regime") is not None})
    for regime in regimes:
        rows = [row for row in seed_rows if row.get("regime") == regime]
        completed = [row for row in rows if row["status"] == "complete"]
        status = "pending"
        if rows and all(row["status"] == "complete" for row in rows):
            status = "complete"
        elif any(row["status"] in {"running", "complete"} for row in rows):
            status = "partial"

        def first_value(key: str) -> Any:
            for row in rows:
                value = row.get(key)
                if value is not None:
                    return value
            return None

        recon_values = [row.get("recon") for row in rows]
        recon_stats = aggregate_numeric(recon_values)
        robust_recon = robust_positive_aggregate(recon_values)

        aggregate_rows.append(
            {
                "regime": regime,
                "status": status,
                "observed_seeds": len(rows),
                "completed_seeds": len(completed),
                "k_min": first_value("k_min"),
                "k_max": first_value("k_max"),
                "kl_normalization": first_value("kl_normalization"),
                "last_epoch_mean": aggregate_numeric([row.get("last_epoch") for row in rows])["mean"],
                "progress_mean": aggregate_numeric([row.get("progress") for row in rows])["mean"],
                "recon_mean": recon_stats["mean"],
                "recon_std": recon_stats["std"],
                "recon_median": robust_recon["median"],
                "recon_stable_mean": robust_recon["stable_mean"],
                "recon_outlier_count": robust_recon["outlier_count"],
                "k_mean": aggregate_numeric([row.get("k_mean") for row in rows])["mean"],
                "k_active_mean": aggregate_numeric([row.get("k_active") for row in rows])["mean"],
                "n_active_frame_mean": aggregate_numeric([row.get("n_active_frame") for row in rows])["mean"],
                "n_active_total_mean": aggregate_numeric([row.get("n_active_total") for row in rows])["mean"],
                "sparsity_mean": aggregate_numeric([row.get("sparsity") for row in rows])["mean"],
                "weighted_kl_gamma_mean": aggregate_numeric([row.get("weighted_kl_gamma") for row in rows])["mean"],
                "weighted_kl_delta_mean": aggregate_numeric([row.get("weighted_kl_delta") for row in rows])["mean"],
                "collapse_rate": aggregate_numeric([1.0 if row.get("collapse") else 0.0 for row in rows])["mean"],
            }
        )
    return seed_rows, aggregate_rows


def summarize_fsdd_runs(repo_root: Path) -> list[dict[str, Any]]:
    entries = [
        ("fsdd_binary_additive_safe", repo_root / "checkpoints/fsdd_binary_additive_safe"),
        ("fsdd_ternary_signed_safe", repo_root / "checkpoints/fsdd_ternary_signed_safe"),
    ]
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in entries:
        row = summarize_training_run_dir(run_dir, dataset="fsdd", run_name=run_name)
        rows.append(row)
    return rows


def write_dashboard(
    path: Path,
    *,
    sinusoid_rows: list[dict[str, Any]],
    sinusoid_energy_rows: list[dict[str, Any]],
    mnist_rows: list[dict[str, Any]],
    fsdd_rows: list[dict[str, Any]],
) -> None:
    sinusoid_fields = [
        "condition",
        "status",
        "completed_seeds",
        "expected_seeds",
        "delta_factorization",
        "presence_estimator",
        "kl_normalization",
        "k_min",
        "support_precision_mean",
        "support_recall_mean",
        "support_f1_mean",
        "topk_f1_mean",
        "amplitude_corr_mean",
        "support_consistency_scaled_mean",
        "gamma_equivariance_error_mean",
        "recon_mse_mean",
        "collapse_rate",
    ]
    sinusoid_energy_fields = [
        "condition",
        "status",
        "model_family",
        "baseline_name",
        "gain_distribution",
        "gain_min",
        "gain_max",
        "normalize_divisor",
        "support_precision_mean",
        "support_recall_mean",
        "support_f1_mean",
        "topk_f1_mean",
        "top1_fourier_corr_mean",
        "high_conf_atom_fraction",
        "n_active_frame_mean",
        "recon_mse_mean",
    ]
    mnist_fields = [
        "regime",
        "status",
        "observed_seeds",
        "completed_seeds",
        "k_min",
        "recon_stable_mean",
        "recon_median",
        "recon_outlier_count",
        "k_mean",
        "n_active_frame_mean",
        "collapse_rate",
    ]
    fsdd_fields = [
        "run_name",
        "status",
        "structure_mode",
        "last_epoch",
        "epochs_target",
        "recon",
        "n_active_frame",
        "n_active_total",
        "collapse",
    ]

    lines = [
        "# Paper Regime Dashboard",
        "",
        "Live tables for the structured sparse paper batch. LibriMix diagnostics remain in `report/tables/librimix_diagnostics.csv` and `report/tables/optimization_ablation.csv`.",
        "",
        "## Sinusoid Multiseed",
        "",
        render_markdown_table(sinusoid_rows, sinusoid_fields).rstrip(),
        "",
    ]
    if sinusoid_energy_rows:
        lines.extend(
            [
                "## Energy Stress",
                "",
                render_markdown_table(sinusoid_energy_rows, sinusoid_energy_fields).rstrip(),
                "",
            ]
        )
    lines.extend(
        [
            "## MNIST Regime",
            "",
            render_markdown_table(mnist_rows, mnist_fields).rstrip(),
            "",
            "## FSDD Case Study",
            "",
            render_markdown_table(fsdd_rows, fsdd_fields).rstrip(),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tables_dir = REPO_ROOT / "report/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    sinusoid_seed_rows: list[dict[str, Any]] = []
    sinusoid_rows: list[dict[str, Any]] = []
    for condition_dir in sorted(path for path in (REPO_ROOT / "results/regime_study").iterdir() if path.is_dir()):
        seed_rows, aggregate_row = summarize_sinusoid_condition_dir(condition_dir)
        sinusoid_seed_rows.extend(seed_rows)
        sinusoid_rows.append(aggregate_row)

    mnist_seed_rows, mnist_rows = summarize_mnist_root(REPO_ROOT / "results/mnist_regime")
    fsdd_rows = summarize_fsdd_runs(REPO_ROOT)
    baseline_compare_conditions = {"HSVAE-lowk-lista", "SCVAE-lista", "SimpleSparseBaseline"}
    sinusoid_energy_rows = [
        row for row in sinusoid_rows if row["condition"] in baseline_compare_conditions or "energy" in row["condition"]
    ]

    write_csv(tables_dir / "sinusoid_regime_seeds.csv", sinusoid_seed_rows)
    write_csv(tables_dir / "sinusoid_regime_multiseed.csv", sinusoid_rows)
    write_csv(tables_dir / "sinusoid_energy_stress.csv", sinusoid_energy_rows)
    write_csv(tables_dir / "mnist_regime_seeds.csv", mnist_seed_rows)
    write_csv(tables_dir / "mnist_regime.csv", mnist_rows)
    write_csv(tables_dir / "fsdd_case_study.csv", fsdd_rows)
    write_dashboard(
        tables_dir / "paper_regime_dashboard.md",
        sinusoid_rows=sinusoid_rows,
        sinusoid_energy_rows=sinusoid_energy_rows,
        mnist_rows=mnist_rows,
        fsdd_rows=fsdd_rows,
    )

    summaries_dir = tables_dir / "paper_batch_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sinusoid_regime_multiseed": sinusoid_rows,
        "sinusoid_energy_stress": sinusoid_energy_rows,
        "sinusoid_regime_seeds": sinusoid_seed_rows,
        "mnist_regime": mnist_rows,
        "mnist_regime_seeds": mnist_seed_rows,
        "fsdd_case_study": fsdd_rows,
    }
    with (summaries_dir / "paper_batch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"Wrote paper-batch tables to {tables_dir}")
    for row in sinusoid_rows:
        print(
            "SINUSOID",
            row["condition"],
            row["status"],
            "support_f1=" + format_float(row["support_f1_mean"]),
            "topk_f1=" + format_float(row["topk_f1_mean"]),
            "recon=" + format_float(row["recon_mse_mean"]),
        )
    for row in mnist_rows:
        print(
            "MNIST",
            row["regime"],
            row["status"],
            "recon_stable=" + format_float(row["recon_stable_mean"]),
            "recon_median=" + format_float(row["recon_median"]),
            "outliers=" + str(row["recon_outlier_count"]),
            "k_mean=" + format_float(row["k_mean"]),
        )
    for row in fsdd_rows:
        print(
            "FSDD",
            row["run_name"],
            row["status"],
            "epoch=" + str(row["last_epoch"]),
            "recon=" + format_float(row["recon"]),
        )


if __name__ == "__main__":
    main()
