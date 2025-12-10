import subprocess
import os
import pandas as pd
import re
import sys
import time
import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)

STUDY_CONFIGS = [
    {
        'name': 'Attn: Sparsemax',
        'attention': 'sparsemax',
        'dynamic_masking': True,
        'inject_noise': True,
        'temporal_aware': True,
        'adaptive_threshold': True
    },
    {
        'name': 'Attn: Entmax15',
        'attention': 'entmax15',
        'dynamic_masking': True,
        'inject_noise': True,
        'temporal_aware': True,
        'adaptive_threshold': True
    }
]

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = Path('/mnt/hdd/gpcc/datasets/cic-ids-2017/tabular')

DATASETS = [
    'cic-ids-2017-dos-hulk.csv'
]

def parse_consolidated_results(output):
    """Parses the consolidated results block from the main script output."""
    consolidated_match = re.search(r"--- Consolidated Results for:.*?---\n(.*?)(?=\n\n={25,})", output, re.DOTALL)
    if not consolidated_match:
        consolidated_match = re.search(r"--- Consolidated Results for:.*?---\n(.*)", output, re.DOTALL)
        if not consolidated_match:
             return None
    
    results_text = consolidated_match.group(1)
    
    def get_metric(metric_name):
        match = re.search(fr"{re.escape(metric_name)}:\s*([nan\d.]+)\s*\+/-\s*([nan\d.]+)", results_text)
        if not match:
            return "N/A"
        
        mean_val = match.group(1)
        ci_val = match.group(2)
        
        if 'nan' in mean_val or 'nan' in ci_val:
            return f"{mean_val} +/- {ci_val}"
            
        return f"{float(mean_val):.4f} +/- {float(ci_val):.4f}"

    return {
        'Test AUC': get_metric("TEST AUC-ROC"),
        'Test AP': get_metric("TEST AP"),
        'Test F1': get_metric("TEST F1-Score"),
        'Total Time (s)': get_metric("Total Time (s)"),
        'Time per Epoch (s)': get_metric("Time per Epoch (s)")
    }


def run_ablation_study():
    """Runs the full ablation study with real-time logging for multiple datasets."""

    for dataset_filename in DATASETS:
        dataset_path = DATASETS_DIR / dataset_filename
        if not dataset_path.exists():
            print(f"\n[ERROR] Dataset file not found at '{dataset_path}'. Skipping dataset.")
            continue

        print(f"\n{'#'*30} STARTING ABLATION STUDY FOR: {dataset_filename} {'#'*30}\n")
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dataset_name = dataset_filename.replace('.csv', '')
        study_dir = Path("results") / f"ablation_final_{dataset_name}_{timestamp}"
        study_dir.mkdir(parents=True, exist_ok=True)
        
        results_file_path = study_dir / "ablation_summary.csv"
        print(f"[INFO] Ablation study results for {dataset_filename} will be saved in: {study_dir}")

        all_results = []
        total_configs = len(STUDY_CONFIGS)

        for i, config in enumerate(STUDY_CONFIGS):
            print(f"\n{'='*25} Running Ablation {i+1}/{total_configs}: {config['name']} on {dataset_filename} {'='*25}")
            
            config_name_safe = re.sub(r'[^a-zA-Z0-9_-]', '', config['name'].replace(' ', '_'))
            config_output_dir = study_dir / config_name_safe
            config_output_dir.mkdir(exist_ok=True)

            run_seed = GLOBAL_SEED + i
            
            cmd = [
                sys.executable, 'main.py',
                '--datasets', str(dataset_path),
                '--attention', config['attention'],
                '--output_dir', str(config_output_dir),
                '--dataset_fraction', '1.0',
                '--seed', str(run_seed)
            ]
            
            if 'anomaly_score' in config:
                cmd.extend(['--anomaly_score', config['anomaly_score']])

            if config.get('dynamic_masking', False):
                cmd.append('--dynamic_masking')
            if config.get('inject_noise', False):
                cmd.append('--inject_noise')
            if config.get('temporal_aware', False):
                cmd.append('--temporal_aware')
            if config.get('adaptive_threshold', False):
                cmd.append('--adaptive_threshold')
            

            print(f"Executing: {' '.join(cmd)}\n")
            
            output_lines = []
            try:
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                      text=True, encoding='utf-8', errors='replace', bufsize=1,
                                      universal_newlines=True, env=env) as process:
                    
                    if process.stdout:
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            output_lines.append(line)
                
                full_output = "".join(output_lines)
                
                if process.returncode != 0:
                    print(f"\n--- ERROR: Subprocess for '{config['name']}' exited with code {process.returncode} ---")
                    continue

                parsed_results = parse_consolidated_results(full_output)
                
                if parsed_results:
                    result_row = {'Configuration': config['name'], **parsed_results}
                    all_results.append(result_row)
                    print("\n--- Ablation Results So Far ---")
                    print(pd.DataFrame(all_results).to_string())
                else:
                    print("\n--- WARNING: Could not parse consolidated results from the output above. ---")

            except Exception as e:
                print(f"\n--- An unexpected script-level ERROR occurred running '{config['name']}': {e} ---")
                continue

        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(results_file_path, index=False)
            print(f"\n{'='*25} Ablation Study for {dataset_filename} Complete {'='*25}")
            print(f"Results saved to {results_file_path}")
            print("\n--- Final Summary ---")
            print(df.to_string())
        else:
            print(f"\nNo results were successfully generated from the ablation study for {dataset_filename}.")

if __name__ == "__main__":
    run_ablation_study()