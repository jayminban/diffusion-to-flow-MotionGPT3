#!/usr/bin/env python3
"""
Generate Pareto plots from benchmark CSV file.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# Hardcoded CSV path
CSV_PATH = "/mnt/data8tb/Documents/models/repo/MotionGPT3/benchmark_results/t2m-val-forward-afew.csv"

# Steps range to display for each config (inclusive)
# Set to None to include all steps
STEPS_RANGE = {
    "flow": (2, 10),       # min_step, max_step
    "diffusion": (2, 20),  # min_step, max_step
}


def find_column(df, keyword):
    """Find column containing keyword with 'mean' but not 'gt'."""
    for col in df.columns:
        if keyword in col and 'mean' in col and 'gt' not in col.lower():
            return col
    return None


def generate_pareto_plots(csv_path, output_dir=None):
    """Generate Pareto plots from benchmark CSV."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # Set output directory
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find metric columns
    fid_col = find_column(df, 'FID')
    r_prec_col = find_column(df, 'R_precision_top_3')
    matching_col = find_column(df, 'Matching_score')

    print(f"\nDetected columns:")
    print(f"  FID: {fid_col}")
    print(f"  R-Precision: {r_prec_col}")
    print(f"  Matching Score: {matching_col}")

    plot_configs = [
        (fid_col, 'FID Score (lower is better)', 'fid'),
        (r_prec_col, 'R-Precision@3 (higher is better)', 'r_precision'),
        (matching_col, 'Matching Score (lower is better)', 'matching_score'),
    ]

    generated_plots = []

    for y_col, y_label, plot_name in plot_configs:
        if y_col is None:
            print(f"Warning: Could not find column for {plot_name}, skipping...")
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for config in df['config'].unique():
            subset = df[df['config'] == config].sort_values('sampling_steps')

            # Filter by steps range if specified
            if config in STEPS_RANGE and STEPS_RANGE[config] is not None:
                min_step, max_step = STEPS_RANGE[config]
                subset = subset[(subset['sampling_steps'] >= min_step) &
                               (subset['sampling_steps'] <= max_step)]

            if len(subset) == 0:
                continue

            ax.plot(subset['mean_time_ms'], subset[y_col], 'o-', label=config, markersize=8)
            for _, row in subset.iterrows():
                ax.annotate(f"{int(row['sampling_steps'])}",
                           (row['mean_time_ms'], row[y_col]),
                           textcoords="offset points", xytext=(5, 5), fontsize=9)

        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel(y_label)
        ax.set_title(f'Inference Time vs {plot_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        pdf_path = output_dir / f"pareto_{plot_name}_{timestamp}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {pdf_path}")
        generated_plots.append(pdf_path)

    return generated_plots


def main():
    generate_pareto_plots(CSV_PATH)
    print("\nDone!")


if __name__ == "__main__":
    main()
