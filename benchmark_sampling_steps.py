#!/usr/bin/env python3
"""
Benchmark script for measuring inference time and test metrics across different sampling steps.
- Timing: Uses training data prompts with val_t2m_forward for pure inference timing
- Metrics: Uses test.py style trainer.test() for evaluation

Usage:
    python benchmark_sampling_steps.py

Output:
    - benchmark_results/benchmark_results_<timestamp>.json: Full detailed results
    - benchmark_results/benchmark_pareto_<timestamp>.csv: CSV for Pareto plotting
"""

import json
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Register the eval resolver once at module load (before parse_args is called)
try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass  # Already registered

from motGPT.callback import build_callbacks
from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae
from motGPT.utils.logger import create_logger
from motGPT.diffusion import create_diffusion


# =============================================================================
# Configuration
# =============================================================================

# Configs to benchmark
CONFIGS = {
    "flow": {
        "path": "./configs/MoT_vae_stage1_t2m-flow.yaml",
        "sampling_steps_key": "sampling_steps",
        "steps_to_test": [2,3,4,5,6,7,8,9,10],
    },
    "diffusion": {
        "path": "./configs/MoT_vae_stage1_t2m-diffusion.yaml",
        "sampling_steps_key": "diffusion_sampling_steps",
        "steps_to_test": [5,6,7,8,9,10,11,12,13,14,15],
    },
}

# Training data path for loading prompts
TRAIN_DATA_PATH = "/mnt/data8tb/Documents/project/capstone-motion/HumanML3D/HumanML3D"

# Benchmark settings
WARMUP_RUNS = 3
TIMED_RUNS = 10
TIMING_BATCH_SIZE = 32
REPLICATION_TIMES = 1

# What to run
RUN_TIMING = True      # Measure inference time
RUN_EVALUATION = True  # Run benchmark metrics (FID, R-Precision, etc.)

# Precomputed scores (used when RUN_EVALUATION is False)
PRECOMPUTED_SCORES_PATH = "./benchmark_results/precomputed_scores.csv"


# =============================================================================
# Utility Functions
# =============================================================================

def load_precomputed_scores(csv_path):
    """Load precomputed benchmark scores from CSV file."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    scores = {}
    for _, row in df.iterrows():
        key = (row['config'], int(row['sampling_steps']))
        scores[key] = {
            'Metrics/FID/mean': row['Metrics/FID/mean'],
            'Metrics/R_precision_top_3/mean': row['Metrics/R_precision_top_3/mean'],
            'Metrics/Matching_score/mean': row['Metrics/Matching_score/mean'],
            'Metrics/Diversity/mean': row['Metrics/Diversity/mean'],
        }
    return scores


def load_training_prompts(data_path, num_samples=100, seed=42):
    """
    Load prompts from training data with deterministic selection.
    """
    random.seed(seed)
    np.random.seed(seed)

    train_file = Path(data_path) / "train.txt"
    texts_dir = Path(data_path) / "texts"

    # Read training sample IDs
    with open(train_file, 'r') as f:
        sample_ids = [line.strip() for line in f.readlines()]

    # Deterministically select samples
    selected_ids = sample_ids[:num_samples]

    prompts = []
    lengths = []

    for sample_id in selected_ids:
        text_file = texts_dir / f"{sample_id}.txt"
        if text_file.exists():
            with open(text_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Parse the first line: text#pos_tags#start#end
                    first_line = lines[0].strip()
                    text = first_line.split('#')[0]
                    prompts.append(text)
                    # Use maximum frame length for consistent timing
                    lengths.append(200)

    return prompts, lengths


def get_metric_statistics(values, replication_times):
    """Calculate mean and confidence interval."""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def set_sampling_steps(model, steps, is_flow=True):
    """Set sampling steps for the model at runtime."""
    if is_flow:
        model.lm.sampling_steps = steps
        print(f"[Benchmark] Set flow sampling_steps = {steps}")
    else:
        diffloss = model.lm.diffloss
        diffloss.gen_diffusion = create_diffusion(
            timestep_respacing=str(steps),
            noise_schedule='linear',
            use_kl=False,
            learn_sigma=False,
            sigma_small=True
        )
        print(f"[Benchmark] Set diffusion sampling_steps = {steps}")


class MockTrainer:
    """Mock trainer for val_t2m_forward that needs self.trainer.datamodule.is_mm"""
    def __init__(self, datamodule):
        self.datamodule = datamodule


def measure_inference_time(model, datamodule, device, prompts, lengths,
                           batch_size=32, num_warmup=3, num_timed=10):
    """
    Measure inference time for both:
    1. End-to-End Inference (val_t2m_forward): LLM + motion prior + VAE decode
    2. Motion prior only (sample_tokens): just the denoising loop
    """
    model.eval()

    # Setup mock trainer
    datamodule.is_mm = False
    mock_trainer = MockTrainer(datamodule)
    model._trainer = mock_trainer
    model.datamodule = datamodule

    # Create batches from prompts
    num_batches = min(len(prompts) // batch_size, num_warmup + num_timed)
    if num_batches < num_warmup + num_timed:
        print(f"  Warning: Only {num_batches} batches available, need {num_warmup + num_timed}")

    full_times = []
    prior_times = []

    with torch.no_grad():
        batch_idx = 0

        # Warmup runs (end-to-end inference)
        print(f"  Running {num_warmup} warmup iterations...")
        for i in range(num_warmup):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_texts = prompts[start_idx:end_idx]
            batch_lengths = lengths[start_idx:end_idx]

            max_len = max(batch_lengths)
            motion_placeholder = torch.zeros(len(batch_texts), max_len, 263).to(device)

            batch = {
                'text': batch_texts,
                'length': batch_lengths,
                'motion': motion_placeholder,
            }

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            _ = model.val_t2m_forward(batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            batch_idx += 1

        # Timed runs - measure both end-to-end inference and sample_tokens
        print(f"  Running {num_timed} timed iterations...")
        for i in range(num_timed):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # Wrap around if needed
            if end_idx > len(prompts):
                batch_idx = 0
                start_idx = 0
                end_idx = batch_size

            batch_texts = prompts[start_idx:end_idx]
            batch_lengths = lengths[start_idx:end_idx]

            # === Measure END-TO-END INFERENCE (val_t2m_forward) ===
            max_len = max(batch_lengths)
            motion_placeholder = torch.zeros(len(batch_texts), max_len, 263).to(device)

            batch = {
                'text': batch_texts,
                'length': batch_lengths,
                'motion': motion_placeholder,
            }

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model.val_t2m_forward(batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            full_elapsed_ms = (end_time - start_time) * 1000
            full_times.append(full_elapsed_ms)

            # === Measure SAMPLE_TOKENS ONLY ===
            # LLM forward (not timed)
            outputs = model.lm.generate_conditional(
                batch_texts,
                lengths=batch_lengths,
                stage='test',
                tasks=None,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            _ = model.lm.sample_tokens(
                outputs, device,
                temperature=1.0, cfg=model.guidance_scale,
                vae_mean_std_inv=model.vae.mean_std_inv
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            prior_elapsed_ms = (end_time - start_time) * 1000
            prior_times.append(prior_elapsed_ms)

            print(f"    Run {i+1}/{num_timed}: full={full_elapsed_ms:.2f}ms, prior={prior_elapsed_ms:.2f}ms")

            batch_idx += 1

    full_times = np.array(full_times)
    prior_times = np.array(prior_times)

    return {
        # End-to-End Inference timing (val_t2m_forward)
        'full_mean_time_ms': float(np.mean(full_times)),
        'full_std_time_ms': float(np.std(full_times)),
        'full_min_time_ms': float(np.min(full_times)),
        'full_max_time_ms': float(np.max(full_times)),
        'full_samples_per_sec': float((batch_size / np.mean(full_times)) * 1000),
        # Motion prior only timing (sample_tokens)
        'prior_mean_time_ms': float(np.mean(prior_times)),
        'prior_std_time_ms': float(np.std(prior_times)),
        'prior_min_time_ms': float(np.min(prior_times)),
        'prior_max_time_ms': float(np.max(prior_times)),
        'prior_samples_per_sec': float((batch_size / np.mean(prior_times)) * 1000),
        # Common
        'batch_size': batch_size,
        'num_timed_runs': num_timed,
    }


def run_evaluation(trainer, model, datamodule, cfg, replication_times=1):
    """Run evaluation exactly like test.py using trainer.test()."""
    all_metrics = {}

    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        print(f"  Evaluating {metrics_type} - Replication {i}")

        metrics = trainer.test(model, datamodule=datamodule)[0]

        if "TM2TMetrics" in metrics_type and cfg.model.params.task == "t2m" and cfg.model.params.stage != 'vae':
            print(f"  Evaluating MultiModality - Replication {i}")
            datamodule.mm_mode(True)
            mm_metrics = trainer.test(model, datamodule=datamodule)[0]
            metrics.update(mm_metrics)
            datamodule.mm_mode(False)

        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key].append(item)

    result_metrics = {}
    for key, values in all_metrics.items():
        if 'epoch' in key or key == 'task':
            continue
        mean, conf = get_metric_statistics(np.array(values), replication_times)
        result_metrics[f"{key}/mean"] = float(mean)
        result_metrics[f"{key}/conf"] = float(conf)

    return result_metrics


def benchmark_single_config(config_name, config_info, prompts, lengths, precomputed_scores=None):
    """Benchmark a single config across different sampling steps."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"Config: {config_info['path']}")
    print(f"{'='*60}")

    results = []
    is_flow = config_name == "flow"

    for steps in config_info['steps_to_test']:
        print(f"\n--- Testing {steps} sampling steps ---")

        sys.argv = ['benchmark', '--cfg', config_info['path']]
        cfg = parse_args(phase="test")
        cfg.FOLDER = cfg.TEST.FOLDER

        logger = create_logger(cfg, phase="test")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        callbacks = build_callbacks(cfg, logger=logger, phase="test")
        datamodule = build_data(cfg)
        model = build_model(cfg, datamodule).eval()

        if cfg.ACCELERATOR == "gpu":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if cfg.TRAIN.PRETRAINED_VAE:
            load_pretrained_vae(cfg, model, logger)

        if cfg.TEST.CHECKPOINTS:
            state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu", weights_only=False)["state_dict"]
            model.load_state_dict(state_dict)

        model.to(device)
        set_sampling_steps(model, steps, is_flow=is_flow)
        pl.seed_everything(cfg.SEED_VALUE)

        # === TIMING: Use training prompts with val_t2m_forward ===
        timing_results = {}
        if RUN_TIMING:
            print("Measuring inference time...")
            timing_results = measure_inference_time(
                model, datamodule, device, prompts, lengths,
                batch_size=TIMING_BATCH_SIZE,
                num_warmup=WARMUP_RUNS,
                num_timed=TIMED_RUNS
            )

        # === METRICS: Use test.py style trainer.test() or precomputed scores ===
        eval_metrics = {}
        trainer = None
        if RUN_EVALUATION:
            print("Running evaluation...")

            if cfg.ACCELERATOR == 'cpu':
                devices = cfg.DEVICE if isinstance(cfg.DEVICE, int) else 1
            else:
                devices = list(range(len(cfg.DEVICE)))

            trainer = pl.Trainer(
                benchmark=False,
                max_epochs=cfg.TRAIN.END_EPOCH,
                accelerator=cfg.ACCELERATOR,
                devices=devices,
                default_root_dir=cfg.FOLDER_EXP,
                reload_dataloaders_every_n_epochs=1,
                deterministic=False,
                detect_anomaly=False,
                enable_progress_bar=True,
                logger=None,
                callbacks=callbacks,
            )

            eval_metrics = run_evaluation(trainer, model, datamodule, cfg, replication_times=REPLICATION_TIMES)
        else:
            # Use precomputed scores if available
            if precomputed_scores and (config_name, steps) in precomputed_scores:
                eval_metrics = precomputed_scores[(config_name, steps)]
                print(f"  Using precomputed scores for {config_name} {steps} steps")

        result = {
            'config': config_name,
            'sampling_steps': steps,
            **timing_results,
            **eval_metrics,
        }
        results.append(result)

        # Print summary
        print(f"\nResults for {steps} steps:")
        if RUN_TIMING:
            print(f"  End-to-End:     {timing_results['full_mean_time_ms']:.2f} ± {timing_results['full_std_time_ms']:.2f} ms")
            print(f"  Motion prior:   {timing_results['prior_mean_time_ms']:.2f} ± {timing_results['prior_std_time_ms']:.2f} ms")
            print(f"  Throughput:     {timing_results['full_samples_per_sec']:.2f} samples/sec")
        if RUN_EVALUATION:
            fid = None
            r_prec = None
            for k, v in eval_metrics.items():
                if 'FID' in k and 'mean' in k and 'gt' not in k.lower():
                    fid = v
                if 'R_precision_top_3' in k and 'mean' in k and 'gt' not in k.lower():
                    r_prec = v
            print(f"  FID: {fid:.4f}" if fid is not None else "  FID: N/A")
            print(f"  R-Precision@3: {r_prec:.4f}" if r_prec is not None else "  R-Precision@3: N/A")

        del model, datamodule
        if trainer is not None:
            del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def generate_plots_from_dataframe(df, output_dir, timestamp):
    """Generate Pareto plots from a dataframe."""
    # Set publication-quality defaults
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    # Define colors
    colors = {
        'diffusion': '#1f77b4',  # Blue
        'flow': '#2ca02c',       # Green
    }

    def find_column(df, keyword):
        for col in df.columns:
            if keyword in col and 'mean' in col and 'gt' not in col.lower():
                return col
        return None

    fid_col = find_column(df, 'FID')
    r_prec_col = find_column(df, 'R_precision_top_3')
    matching_col = find_column(df, 'Matching_score')

    metric_configs = [
        (fid_col, 'FID Score', 'fid'),
        (r_prec_col, 'R-Precision@3', 'r_precision'),
        (matching_col, 'Matching Score', 'matching_score'),
    ]

    # Two sets of plots: end-to-end inference and motion prior only
    timing_configs = [
        ('full_mean_time_ms', 'End-to-End Inference', 'full'),
        ('prior_mean_time_ms', 'Motion Prior Only Inference', 'prior'),
    ]

    generated_plots = []

    for time_col, time_label, time_name in timing_configs:
        if time_col not in df.columns:
            continue

        for y_col, y_label, metric_name in metric_configs:
            if y_col is None or y_col not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))

            for config in df['config'].unique():
                subset = df[df['config'] == config].sort_values('sampling_steps')
                color = colors.get(config, '#333333')
                ax.plot(subset[time_col], subset[y_col], 'o-',
                       label=config.capitalize(), color=color, markersize=6, linewidth=1.8)
                for _, row in subset.iterrows():
                    ax.annotate(f"{int(row['sampling_steps'])}",
                               (row[time_col], row[y_col]),
                               textcoords="offset points", xytext=(5, 5), fontsize=9)

            ax.set_xlabel(f'Inference Time (ms)')
            ax.set_ylabel(y_label)
            ax.set_title(f'{time_label}: Time vs {metric_name.replace("_", " ").title()}')
            ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(direction='in', which='both')

            pdf_path = output_dir / f"benchmark_{time_name}_{metric_name}_{timestamp}.pdf"
            fig.savefig(pdf_path, format='pdf')
            plt.close(fig)
            print(f"Plot saved to: {pdf_path}")
            generated_plots.append(pdf_path)

    return generated_plots


def main():
    """Main benchmark function."""
    print("=" * 60)
    print("MotionGPT3 Sampling Steps Benchmark")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    output_dir = Path("./benchmark_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # === PLOT-ONLY MODE: When both RUN_TIMING and RUN_EVALUATION are False ===
    if not RUN_TIMING and not RUN_EVALUATION:
        print("\n[PLOT-ONLY MODE] Both RUN_TIMING and RUN_EVALUATION are False.")
        print(f"Loading data from: {PRECOMPUTED_SCORES_PATH}")

        # Error if precomputed scores path not set
        if not PRECOMPUTED_SCORES_PATH:
            raise ValueError("PRECOMPUTED_SCORES_PATH must be set when both RUN_TIMING and RUN_EVALUATION are False")

        # Error if file doesn't exist
        precomputed_path = Path(PRECOMPUTED_SCORES_PATH)
        if not precomputed_path.exists():
            raise FileNotFoundError(f"Precomputed scores file not found: {PRECOMPUTED_SCORES_PATH}")

        # Load the CSV
        df = pd.read_csv(precomputed_path)
        print(f"Loaded {len(df)} rows from {PRECOMPUTED_SCORES_PATH}")
        print(f"Columns: {list(df.columns)}")

        # Validate required columns exist
        required_columns = ['config', 'sampling_steps', 'full_mean_time_ms', 'prior_mean_time_ms']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in precomputed scores: {missing_columns}")

        # Check for at least one metric column
        metric_keywords = ['FID', 'R_precision', 'Matching_score']
        has_metric = any(
            any(kw in col for kw in metric_keywords)
            for col in df.columns
        )
        if not has_metric:
            raise ValueError("Precomputed scores must contain at least one metric column (FID, R_precision, or Matching_score)")

        # Filter data based on steps_to_test in CONFIGS
        filtered_rows = []
        for config_name, config_info in CONFIGS.items():
            steps_to_test = config_info.get('steps_to_test', [])
            config_data = df[(df['config'] == config_name) & (df['sampling_steps'].isin(steps_to_test))]
            filtered_rows.append(config_data)

        if filtered_rows:
            df = pd.concat(filtered_rows, ignore_index=True)
            print(f"Filtered to {len(df)} rows based on CONFIGS steps_to_test")

        # Generate plots directly from precomputed data
        generate_plots_from_dataframe(df, output_dir, timestamp)

        print(f"\n[PLOT-ONLY MODE] Done!")
        return df.to_dict('records')

    # === NORMAL MODE: Run timing and/or evaluation ===

    # Load training prompts once
    print(f"\nLoading training prompts from {TRAIN_DATA_PATH}...")
    prompts, lengths = load_training_prompts(
        TRAIN_DATA_PATH,
        num_samples=500,  # Load enough for multiple batches
        seed=42
    )
    print(f"Loaded {len(prompts)} prompts")

    # Load precomputed scores if not running evaluation
    precomputed_scores = None
    if not RUN_EVALUATION and PRECOMPUTED_SCORES_PATH:
        try:
            precomputed_scores = load_precomputed_scores(PRECOMPUTED_SCORES_PATH)
            print(f"Loaded precomputed scores from {PRECOMPUTED_SCORES_PATH}")
        except Exception as e:
            print(f"Warning: Could not load precomputed scores: {e}")

    all_results = []

    for config_name, config_info in CONFIGS.items():
        try:
            results = benchmark_single_config(config_name, config_info, prompts, lengths, precomputed_scores)
            all_results.extend(results)
        except Exception as e:
            print(f"Error benchmarking {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    df = pd.DataFrame(all_results)
    pareto_columns = ['config', 'sampling_steps',
                      'full_mean_time_ms', 'full_std_time_ms', 'full_samples_per_sec',
                      'prior_mean_time_ms', 'prior_std_time_ms', 'prior_samples_per_sec']
    for col in df.columns:
        if any(m in col for m in ['FID', 'R_precision', 'Matching_score', 'Diversity', 'MultiModality']):
            if '/mean' in col and 'gt' not in col.lower():
                pareto_columns.append(col)

    pareto_df = df[[c for c in pareto_columns if c in df.columns]]
    csv_path = output_dir / f"benchmark_pareto_{timestamp}.csv"
    pareto_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Generate plots
    generate_plots_from_dataframe(df, output_dir, timestamp)

    return all_results


if __name__ == "__main__":
    main()
