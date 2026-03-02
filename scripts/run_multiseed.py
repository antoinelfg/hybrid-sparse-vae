#!/usr/bin/env python
"""Run training across multiple seeds and aggregate results.

This script sequentially runs `train.py` with multiple seeds, collects
the final performance metrics, and outputs a summary table with mean ± std.

Usage:
  python scripts/run_multiseed.py --config-name=core4_champ dict_init=random ...
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SEEDS = [42, 123, 456, 789, 1337]


def run_training(seed: int, hydra_args: list[str]) -> Path:
    """Run train.py for a specific seed and return the result directoy."""
    # Construct command
    cmd = [sys.executable, str(REPO_ROOT / "train.py"), f"seed={seed}"] + hydra_args
    print(f"\n[{seed}] Running: {' '.join(cmd)}")
    
    # We need to capture the hydra output directory from the logs if possible
    # But hydra usually creates outputs/YYYY-MM-DD/HH-MM-SS. For a simpler
    # approach, we'll let train.py run and assume the user checks the latest
    # or we can pass a specific output dir via hydra overrides.
    
    # To reliably track results, we'll force the hydra run dir
    # hydra.run.dir=results/multiseed_champion/seed_${seed}
    run_dir = REPO_ROOT / f"results/multiseed_champion/seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = run_dir / "train.log"
    if log_path.exists():
        print(f"[{seed}] Run already exists at {run_dir}. Skipping training.")
        return run_dir
    
    cmd.append(f"hydra.run.dir={run_dir}")
    cmd.append(f"save_dir={run_dir}")
    
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[{seed}] ERROR: Training failed with return code {result.returncode}")
        return None
        
    return run_dir


def extract_metrics(log_path: Path) -> dict[str, float]:
    """Extract the final epoch metrics from a train.log file."""
    if not log_path.exists():
        return None
        
    final_metrics = {}
    import re
    # Match pattern: Epoch  NNN [PHASE] | recon 1.234 | kl_γ 9.9 | ...
    pattern = re.compile(
        r"Epoch\s+(\d+).*?"
        r"recon\s+([\d.]+).*?"
        r"kl_γ\s+([\d.]+).*?"
        r"kl_δ\s+([\d.]+).*?"
        r"k̄=([\d.]+).*?"
        r"n_act=([\d.]+).*?"
        r"δ₀=([\d.]+)%.*?"
        r"Δdict=([\d.]+)"
    )
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                # Always overwrite so we keep the last epoch's values
                final_metrics = {
                    "recon": float(m.group(2)),
                    "kl_gamma": float(m.group(3)),
                    "kl_delta": float(m.group(4)),
                    "k_mean": float(m.group(5)),
                    "n_active": float(m.group(6)),
                    "sparsity": float(m.group(7)) / 100.0,
                    "dict_drift": float(m.group(8))
                }
    
    return final_metrics if final_metrics else None


def main():
    # Parse basic args, pass the rest to Hydra
    parser = argparse.ArgumentParser(description="Multi-seed training runner")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated list of seeds")
    parser.add_argument("--output-dir", type=str, default="results/multiseed_summary",
                        help="Where to save the aggregated JSON")
    
    args, hydra_args = parser.parse_known_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting multi-seed run over {len(seeds)} seeds: {seeds}")
    if hydra_args:
        print(f"Hydra overrides: {' '.join(hydra_args)}")
    
    all_results = {}
    
    for seed in seeds:
        run_dir = run_training(seed, hydra_args)
        if run_dir:
            log_path = run_dir / "train.log"
            metrics = extract_metrics(log_path)
            if metrics:
                all_results[seed] = metrics
                print(f"[{seed}] Completed. Recon MSE: {metrics['recon']:.4f}, Sparsity: {metrics['sparsity']:.2%}")
            else:
                print(f"[{seed}] Failed to extract metrics from {log_path}")
    
    if not all_results:
        print("No successful runs completed.")
        sys.exit(1)
        
    # Aggregate metrics
    metrics_keys = list(next(iter(all_results.values())).keys())
    agg_results = {}
    
    for key in metrics_keys:
        values = [res[key] for res in all_results.values()]
        agg_results[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values
        }
        
    # Print summary table
    print("\n" + "="*60)
    print(" MULTI-SEED AGGREGATED RESULTS ")
    print("="*60)
    print(f"{'Metric':<15} | {'Mean ± Std':<20} | {'Median':<10} | {'Min / Max'}")
    print("-" * 60)
    for key in ["recon", "sparsity", "n_active", "k_mean", "dict_drift", "kl_gamma"]:
        if key in agg_results:
            stats = agg_results[key]
            mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            if key == "sparsity":
                mean_std = f"{stats['mean']:.2%} ± {stats['std']:.2%}"
            format_min_max = f"{stats['min']:.4f} / {stats['max']:.4f}"
            print(f"{key:<15} | {mean_std:<20} | {stats['median']:<10.4f} | {format_min_max}")
    print("="*60)
    
    # Save to JSON
    summary_path = out_dir / "multiseed_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "seeds": seeds,
            "hydra_args": hydra_args,
            "aggregated": agg_results,
            "individual": all_results
        }, f, indent=2)
        
    print(f"\nDetailed summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
