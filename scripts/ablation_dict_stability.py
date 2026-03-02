#!/usr/bin/env python
"""Ablation study for random dictionary initialization stability.

Tests various warmup and gradient clipping strategies to find the
configuration that maximizes the percentage of seeds landing in a
good basin (recon MSE < 1.5).

Usage:
  python scripts/ablation_dict_stability.py
"""

import sys
import json
from pathlib import Path
import numpy as np

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import helpers from run_multiseed
from scripts.run_multiseed import run_training, extract_metrics, DEFAULT_SEEDS

CONFIGS_TO_TEST = {
    "baseline": [
        "dict_init=random",
        "dict_lr_mult=0.1"
    ],
    "freeze_200": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "dict_warmup_epochs=200"
    ],
    "lr_warmup": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "dict_lr_warmup=True"
    ],
    "grad_clip_0.5": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "gradient_clip_dict=0.5"
    ],
    "freeze_200_clip_0.5": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "dict_warmup_epochs=200",
        "gradient_clip_dict=0.5"
    ],
    "lr_warmup_clip_0.5": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "dict_lr_warmup=True",
        "gradient_clip_dict=0.5"
    ],
}


import argparse

def main():
    parser = argparse.ArgumentParser(description="Dict Stability Ablation")
    parser.add_argument("--task-id", type=int, default=None,
                        help="SLURM array task ID (0 to N*M-1) to run a single configuration/seed.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "ablation_dict_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the flat list of (config_name, hydra_args, seed)
    tasks = []
    for config_name, hydra_args in CONFIGS_TO_TEST.items():
        for seed in DEFAULT_SEEDS:
            tasks.append((config_name, hydra_args, seed))
            
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(tasks):
            print(f"ERROR: Task ID {args.task_id} out of bounds (0-{len(tasks)-1})")
            sys.exit(1)
            
        config_name, hydra_args, seed = tasks[args.task_id]
        print(f"--- SLURM TASK {args.task_id} ---")
        print(f"Config: {config_name}")
        print(f"Seed: {seed}")
        
        run_dir = out_dir / config_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [sys.executable, str(REPO_ROOT / "train.py"), f"seed={seed}"] + hydra_args
        cmd.append(f"hydra.run.dir={run_dir}")
        cmd.append(f"save_dir={run_dir}")
        
        import subprocess
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        sys.exit(result.returncode)

    print(f"Starting dictionary stability ablation over {len(CONFIGS_TO_TEST)} configs.")
    print(f"Seeds per config: {DEFAULT_SEEDS}")
    print("Goal: Maximize % of seeds with recon MSE < 1.5")
    print("=" * 60)
    
    results_summary = {}
    
    for config_name, hydra_args in CONFIGS_TO_TEST.items():
        print(f"\n--- Testing Config: {config_name} ---")
        
        config_results = []
        for seed in DEFAULT_SEEDS:
            run_dir = out_dir / config_name / f"seed_{seed}"
            
            # If the user ran them via SLURM array, just extract the existing metrics!
            log_path = run_dir / "train.log"
            if not log_path.exists():
                # Run it sequentially if it hasn't been run yet
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(REPO_ROOT / "train.py"), f"seed={seed}"] + hydra_args
                cmd.append(f"hydra.run.dir={run_dir}")
                cmd.append(f"save_dir={run_dir}")
                
                import subprocess
                result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True)
                
                if result.returncode != 0:
                    print(f"  [{seed}] Failed!")
                    continue
                
            metrics = extract_metrics(log_path)
            
            if metrics:
                recon = metrics["recon"]
                is_good = recon < 1.5
                print(f"  [{seed}] Recon: {recon:.3f} | Good Basin: {'✅' if is_good else '❌'}")
                config_results.append({
                    "seed": seed,
                    "recon": recon,
                    "is_good": is_good,
                    "metrics": metrics
                })
            else:
                print(f"  [{seed}] Could not extract metrics.")
                
        if not config_results:
            print(f"No results for {config_name}")
            continue
            
        # Aggregate
        recons = [r["recon"] for r in config_results]
        good_count = sum(r["is_good"] for r in config_results)
        success_rate = good_count / len(config_results)
        
        results_summary[config_name] = {
            "success_rate": success_rate,
            "mean_recon": float(np.mean(recons)),
            "median_recon": float(np.median(recons)),
            "min_recon": float(np.min(recons)),
            "max_recon": float(np.max(recons)),
            "details": config_results
        }
        
    print("\n" + "=" * 80)
    print(" ABLATION SUMMARY: DICTIONARY INIT STABILITY")
    print("=" * 80)
    print(f"{'Config Name':<25} | {'Success Rate':<15} | {'Median Recon':<15} | {'Mean Recon'}")
    print("-" * 80)
    
    # Sort by success rate (descending) then median recon (ascending)
    sorted_configs = sorted(
        results_summary.keys(),
        key=lambda k: (-results_summary[k]["success_rate"], results_summary[k]["median_recon"])
    )
    
    for name in sorted_configs:
        stats = results_summary[name]
        sr_str = f"{stats['success_rate']:.0%} ({int(stats['success_rate']*len(DEFAULT_SEEDS))}/{len(DEFAULT_SEEDS)})"
        print(f"{name:<25} | {sr_str:<15} | {stats['median_recon']:<15.3f} | {stats['mean_recon']:.3f}")
        
    print("=" * 80)
    
    summary_path = out_dir / "stability_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved detailed results to {summary_path}")


if __name__ == "__main__":
    main()
