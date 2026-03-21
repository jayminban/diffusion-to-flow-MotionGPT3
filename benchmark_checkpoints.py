#!/usr/bin/env python
"""
Benchmark multiple checkpoints and store results in SQLite database.
Edit the CONFIG section below to change settings.
"""
from datetime import datetime

# =============================================================================
# CONFIG - Edit these values
# =============================================================================

# CONFIG_PATH = "configs/MoT_vae_stage1_t2m-flow.yaml"
CONFIG_PATH = "configs/MoT_vae_stage1_t2m-diffusion.yaml"
CONFIG_ASSETS = "configs/assets.yaml"
DB_PATH = "results/" + datetime.now().strftime("%m-%d_%H%M") + "-results.sqlite3"
GROUP_NAME = "Flow-every-checkpoint-200epochs"

# Option 1: Specify a directory - will benchmark all .ckpt files inside
CHECKPOINT_DIR = "/mnt/data8tb/Documents/models/repo/MotionGPT3/experiments/motgpt_2optimizer/Stage-1-diffusion-save-every-epoch-success/checkpoints"  # e.g., "experiments/motgpt/MoT_vae_stage1_t2m-flow/checkpoints"
# CHECKPOINT_DIR = None  # e.g., "experiments/motgpt/MoT_vae_stage1_t2m-flow/checkpoints"

# Option 2: Specify individual checkpoints (used if CHECKPOINT_DIR is empty)
CHECKPOINTS = [
    # "/mnt/data8tb/Documents/models/repo/MotionGPT3/experiments/motgpt_2optimizer/Stage-1-flow-save-every-epoch-success/checkpoints/min-FID-epoch=epoch=46.ckpt",
    # "/mnt/data8tb/Documents/models/repo/MotionGPT3/experiments/motgpt_2optimizer/Stage-1-flow-save-every-epoch-success/checkpoints/epoch=194.ckpt",
    # Add more checkpoints here...
]

# =============================================================================
# END CONFIG
# =============================================================================

import sqlite3
import json
import subprocess
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
import tempfile


def create_tables(conn):
    """Create tables if they don't exist."""
    cursor = conn.cursor()

    # Individual checkpoint results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS checkpoint_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            group_name TEXT,
            checkpoint_path TEXT,
            epoch INTEGER,
            config_path TEXT,
            sampling_steps INTEGER,
            use_rk4 INTEGER,
            cfg_scale REAL,
            fid REAL,
            fid_conf REAL,
            r1 REAL,
            r1_conf REAL,
            r2 REAL,
            r2_conf REAL,
            r3 REAL,
            r3_conf REAL,
            matching_score REAL,
            matching_conf REAL,
            mm_dist REAL,
            mm_dist_conf REAL,
            diversity REAL,
            diversity_conf REAL,
            multimodality REAL,
            multimodality_conf REAL,
            gt_r1 REAL,
            gt_r2 REAL,
            gt_r3 REAL,
            gt_matching_score REAL,
            gt_diversity REAL
        )
    ''')

    # Group summary statistics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS group_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            group_name TEXT,
            num_checkpoints INTEGER,
            mean_fid REAL,
            std_fid REAL,
            mean_r1 REAL,
            std_r1 REAL,
            mean_r2 REAL,
            std_r2 REAL,
            mean_r3 REAL,
            std_r3 REAL,
            mean_matching_score REAL,
            std_matching_score REAL,
            best_fid REAL,
            best_fid_checkpoint TEXT,
            best_r3 REAL,
            best_r3_checkpoint TEXT
        )
    ''')

    conn.commit()


def run_benchmark(cfg_path, checkpoint_path, cfg_assets="configs/assets.yaml"):
    """Run test.py with the given checkpoint and return metrics."""
    import re

    # Create temp config with the checkpoint
    cfg = OmegaConf.load(cfg_path)
    cfg.TEST.CHECKPOINTS = str(checkpoint_path)

    # Save temp config
    temp_cfg = Path(tempfile.mktemp(suffix='.yaml'))
    OmegaConf.save(cfg, temp_cfg)

    print(f"\n{'='*60}")
    print(f"Benchmarking: {checkpoint_path}")
    print(f"{'='*60}")

    # Record timestamp before running test.py
    import time
    start_time = time.time()

    try:
        # Run test.py
        result = subprocess.run(
            [sys.executable, "-u", "test.py",
             "--cfg", str(temp_cfg),
             "--cfg_assets", cfg_assets],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Check if test.py failed
        if result.returncode != 0:
            raise RuntimeError(f"test.py failed with return code {result.returncode}")

        # Find metrics files created AFTER we started test.py (no fallback to old files)
        results_dir = Path(__file__).parent / "results"
        metrics_files = list(results_dir.rglob("metrics_*.json"))

        # Filter to only files created after start_time
        new_metrics_files = [f for f in metrics_files if f.stat().st_mtime > start_time]

        if not new_metrics_files:
            raise RuntimeError(
                f"No new metrics file created for {checkpoint_path}. "
                f"test.py may have failed silently. Check output above."
            )

        latest_metrics = max(new_metrics_files, key=lambda p: p.stat().st_mtime)
        print(f"Found metrics file: {latest_metrics}")

        with open(latest_metrics, 'r') as f:
            metrics = json.load(f)

        return metrics

    finally:
        # Clean up temp config
        if temp_cfg.exists():
            temp_cfg.unlink()


def extract_metrics(metrics_dict):
    """Extract key metrics from the raw metrics dictionary."""
    result = {
        'epoch': metrics_dict.get('epoch', -1),
        'fid': None, 'fid_conf': None,
        'r1': None, 'r1_conf': None,
        'r2': None, 'r2_conf': None,
        'r3': None, 'r3_conf': None,
        'matching_score': None, 'matching_conf': None,
        'mm_dist': None, 'mm_dist_conf': None,
        'diversity': None, 'diversity_conf': None,
        'multimodality': None, 'multimodality_conf': None,
        # GT metrics
        'gt_r1': None, 'gt_r2': None, 'gt_r3': None,
        'gt_matching_score': None, 'gt_diversity': None,
    }

    # Map metric names for model metrics
    metric_map = {
        'FID': 'fid',
        'R_precision_top_1': 'r1',
        'R_precision_top_2': 'r2',
        'R_precision_top_3': 'r3',
        'Matching_score': 'matching_score',
        'MM_Dist': 'mm_dist',
        'Diversity': 'diversity',
        'MultiModality': 'multimodality',
    }

    # Map metric names for GT metrics
    gt_metric_map = {
        'R_precision_top_1': 'gt_r1',
        'R_precision_top_2': 'gt_r2',
        'R_precision_top_3': 'gt_r3',
        'Matching_score': 'gt_matching_score',
        'Diversity': 'gt_diversity',
    }

    for key, value in metrics_dict.items():
        if '_GT/' in key:
            # GT metrics
            for metric_name, result_key in gt_metric_map.items():
                if metric_name in key and '/mean' in key:
                    result[result_key] = float(value)
        else:
            # Model metrics
            for metric_name, result_key in metric_map.items():
                if metric_name in key:
                    if '/mean' in key:
                        result[result_key] = float(value)
                    elif '/conf_interval' in key:
                        result[result_key + '_conf'] = float(value)

    return result


def insert_checkpoint_result(conn, group_name, checkpoint_path, config_path,
                             sampling_steps, use_rk4, cfg_scale, metrics):
    """Insert a single checkpoint result into the database."""
    cursor = conn.cursor()

    extracted = extract_metrics(metrics)

    cursor.execute('''
        INSERT INTO checkpoint_results (
            timestamp, group_name, checkpoint_path, epoch, config_path,
            sampling_steps, use_rk4, cfg_scale,
            fid, fid_conf, r1, r1_conf, r2, r2_conf, r3, r3_conf,
            matching_score, matching_conf, mm_dist, mm_dist_conf,
            diversity, diversity_conf, multimodality, multimodality_conf,
            gt_r1, gt_r2, gt_r3, gt_matching_score, gt_diversity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        group_name,
        str(checkpoint_path),
        extracted['epoch'],
        str(config_path),
        sampling_steps,
        1 if use_rk4 else 0,
        cfg_scale,
        extracted['fid'], extracted['fid_conf'],
        extracted['r1'], extracted['r1_conf'],
        extracted['r2'], extracted['r2_conf'],
        extracted['r3'], extracted['r3_conf'],
        extracted['matching_score'], extracted['matching_conf'],
        extracted['mm_dist'], extracted['mm_dist_conf'],
        extracted['diversity'], extracted['diversity_conf'],
        extracted['multimodality'], extracted['multimodality_conf'],
        extracted['gt_r1'], extracted['gt_r2'], extracted['gt_r3'],
        extracted['gt_matching_score'], extracted['gt_diversity'],
    ))

    conn.commit()
    return cursor.lastrowid


def compute_group_summary(conn, group_name):
    """Compute and insert group summary statistics."""
    cursor = conn.cursor()

    # Get all results for this group
    cursor.execute('''
        SELECT checkpoint_path, fid, r1, r2, r3, matching_score
        FROM checkpoint_results
        WHERE group_name = ?
    ''', (group_name,))

    rows = cursor.fetchall()
    if not rows:
        return

    import numpy as np

    checkpoints = [r[0] for r in rows]
    fids = [r[1] for r in rows if r[1] is not None]
    r1s = [r[2] for r in rows if r[2] is not None]
    r2s = [r[3] for r in rows if r[3] is not None]
    r3s = [r[4] for r in rows if r[4] is not None]
    matching_scores = [r[5] for r in rows if r[5] is not None]

    # Find best checkpoints
    best_fid_idx = np.argmin(fids) if fids else 0
    best_r3_idx = np.argmax(r3s) if r3s else 0

    cursor.execute('''
        INSERT INTO group_summary (
            timestamp, group_name, num_checkpoints,
            mean_fid, std_fid, mean_r1, std_r1, mean_r2, std_r2, mean_r3, std_r3,
            mean_matching_score, std_matching_score,
            best_fid, best_fid_checkpoint, best_r3, best_r3_checkpoint
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        group_name,
        len(rows),
        np.mean(fids) if fids else None,
        np.std(fids) if fids else None,
        np.mean(r1s) if r1s else None,
        np.std(r1s) if r1s else None,
        np.mean(r2s) if r2s else None,
        np.std(r2s) if r2s else None,
        np.mean(r3s) if r3s else None,
        np.std(r3s) if r3s else None,
        np.mean(matching_scores) if matching_scores else None,
        np.std(matching_scores) if matching_scores else None,
        min(fids) if fids else None,
        checkpoints[best_fid_idx] if fids else None,
        max(r3s) if r3s else None,
        checkpoints[best_r3_idx] if r3s else None,
    ))

    conn.commit()


def print_results_table(conn, group_name):
    """Print a formatted table of results."""
    cursor = conn.cursor()

    cursor.execute('''
        SELECT checkpoint_path, epoch, fid, r1, r2, r3, matching_score
        FROM checkpoint_results
        WHERE group_name = ?
        ORDER BY id
    ''', (group_name,))

    rows = cursor.fetchall()

    print(f"\n{'='*100}")
    print(f"Results for group: {group_name}")
    print(f"{'='*100}")
    print(f"{'Checkpoint':<50} {'Epoch':>6} {'FID':>8} {'R@1':>8} {'R@2':>8} {'R@3':>8} {'Match':>8}")
    print("-" * 100)

    for row in rows:
        ckpt = Path(row[0]).name[:48]
        epoch = row[1] if row[1] else -1
        fid = f"{row[2]:.4f}" if row[2] else "N/A"
        r1 = f"{row[3]:.4f}" if row[3] else "N/A"
        r2 = f"{row[4]:.4f}" if row[4] else "N/A"
        r3 = f"{row[5]:.4f}" if row[5] else "N/A"
        match = f"{row[6]:.4f}" if row[6] else "N/A"
        print(f"{ckpt:<50} {epoch:>6} {fid:>8} {r1:>8} {r2:>8} {r3:>8} {match:>8}")

    # Print summary
    cursor.execute('''
        SELECT mean_fid, std_fid, mean_r3, std_r3, best_fid, best_r3
        FROM group_summary
        WHERE group_name = ?
        ORDER BY id DESC LIMIT 1
    ''', (group_name,))

    summary = cursor.fetchone()
    if summary:
        print("-" * 100)
        print(f"{'MEAN':<50} {'':<6} {summary[0]:>8.4f} {'':<8} {'':<8} {summary[2]:>8.4f}")
        print(f"{'STD':<50} {'':<6} {summary[1]:>8.4f} {'':<8} {'':<8} {summary[3]:>8.4f}")
        print(f"{'BEST':<50} {'':<6} {summary[4]:>8.4f} {'':<8} {'':<8} {summary[5]:>8.4f}")
    print("=" * 100)


def main():
    # Load config to get sampling params
    cfg = OmegaConf.load(CONFIG_PATH)
    sampling_steps = cfg.lm_ablation.get('sampling_steps', 25)
    use_rk4 = cfg.lm_ablation.get('use_rk4_sampling', False)
    cfg_scale = cfg.lm_ablation.get('model_guidance_scale', 3.0)

    # Get list of checkpoints
    if CHECKPOINT_DIR:
        checkpoint_dir = Path(CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            print(f"Error: Directory not found: {CHECKPOINT_DIR}")
            sys.exit(1)
        checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            print(f"Error: No .ckpt files found in {CHECKPOINT_DIR}")
            sys.exit(1)
        checkpoints = [str(p) for p in checkpoints]
    else:
        checkpoints = CHECKPOINTS

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    print(f"Database: {DB_PATH}")
    print(f"Group: {GROUP_NAME}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Sampling steps: {sampling_steps}, RK4: {use_rk4}, CFG: {cfg_scale}")
    print(f"Checkpoints to benchmark: {len(checkpoints)}")

    # Run benchmarks
    for ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}")
            continue

        metrics = run_benchmark(CONFIG_PATH, ckpt_path, CONFIG_ASSETS)

        if metrics:
            insert_checkpoint_result(
                conn, GROUP_NAME, ckpt_path, CONFIG_PATH,
                sampling_steps, use_rk4, cfg_scale, metrics
            )
            print(f"Results saved for: {ckpt_path}")
        else:
            print(f"Failed to get metrics for: {ckpt_path}")

    # Compute group summary
    compute_group_summary(conn, GROUP_NAME)

    # Print results
    print_results_table(conn, GROUP_NAME)

    conn.close()
    print(f"\nResults saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
