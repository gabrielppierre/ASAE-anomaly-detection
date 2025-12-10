import os
import pandas as pd
import sys
import time
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix
from pyod.models.iforest import IForest
import joblib
import eif as eif

from data import preprocess_data

def set_seed(seed):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)


NUM_RUNS = 5 

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = Path('/mnt/hdd/gpcc/datasets/experiments_cic_ids_18/tabular')

DATASETS = [
    'dataset_benign_and_brute_force_web.csv',
    'dataset_benign_and_brute_force_xss.csv',
    'dataset_benign_and_ddos_attack_hoic.csv',
    'dataset_benign_and_dos_attacks_goldeneye.csv',
    'dataset_benign_and_dos_attacks_hulk.csv',
    'dataset_benign_and_dos_attacks_slowhttptest.csv',
    'dataset_benign_and_dos_attacks_slowloris.csv',
    'dataset_benign_and_ftp-bruteforce.csv',
    'dataset_benign_and_infilteration.csv',
    'dataset_benign_and_sql_injection.csv',
    'dataset_benign_and_ssh-bruteforce.csv'
]

PYOD_MODELS = [
    {'name': 'EIF-IsolationForest', 'model_class': eif.iForest, 'params': {'extension_level': 0}},
    {'name': 'EIF-ExtendedIsolationForest', 'model_class': eif.iForest, 'params': {'extension_level': 'auto'}} # 'auto' will be replaced with num_features - 1
]

def run_pyod_benchmark():
    """Runs a benchmark for PyOD models on specified datasets."""
    
    benchmark_results_dir = Path("pyod_benchmark_results")
    benchmark_results_dir.mkdir(parents=True, exist_ok=True)
    
    all_benchmark_results = []

    for dataset_filename in DATASETS:
        dataset_path = DATASETS_DIR / dataset_filename
        if not dataset_path.exists():
            print(f"\n[ERROR] Dataset file not found at '{dataset_path}'. Skipping dataset.")
            continue

        print(f"\n{'#'*30} STARTING EIF BENCHMARK FOR: {dataset_filename} {'#'*30}\n")
        
        dataset_name = dataset_filename.replace('.csv', '')
        
        dataset_model_results = []

        for model_config in PYOD_MODELS:
            model_name = model_config['name']
            print(f"\n{'='*25} Running {model_name} on {dataset_filename} ({NUM_RUNS} runs) {'='*25}")
            
            run_metrics = {
                'auc_roc': [],
                'ap': [],
                'f1': [],
                'total_time': []
            }

            best_auc_for_model = -1
            best_model_instance = None
            best_run_cm = None

            for run_num in range(NUM_RUNS):
                run_seed = GLOBAL_SEED + run_num
                set_seed(run_seed)
                print(f"  Run {run_num + 1}/{NUM_RUNS} with seed {run_seed}")

                start_time = time.time()
                try:
                    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, _ = preprocess_data(
                        file_path=dataset_path, 
                        scaler_type='minmax', 
                        random_state=run_seed,
                        temporal_aware=True,
                        dataset_fraction=1.0
                    )
                except ValueError as e:
                    print(f"    [ERROR] Data preprocessing failed for {dataset_filename}: {e}. Skipping run.")
                    continue

                X_train_normal = X_train_scaled[y_train == 0] if y_train.sum() > 0 else X_train_scaled
                
                print(f"    Training {model_name} on {len(X_train_normal)} normal samples...")
                

                clf_class = model_config['model_class']
                clf_params = model_config['params'].copy()

                if clf_params.get('extension_level') == 'auto':
                    clf_params['extension_level'] = X_train_scaled.shape[1] - 1

                clf = clf_class(
                    X_train_normal, 
                    ntrees=100, 
                    sample_size=256, 
                    ExtensionLevel=clf_params['extension_level'], 
                    seed=run_seed
                )
                
                y_val_scores = clf.compute_paths(X_val_scaled)
                y_test_scores = clf.compute_paths(X_test_scaled)

                precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_scores)

                if len(thresholds) < len(precisions):
                    thresholds = np.append(thresholds, thresholds[-1] + 1e-6)
                
                f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                                      out=np.zeros_like(precisions), where=(precisions + recalls) != 0)

                best_f1_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[best_f1_idx]
                
                y_test_pred = (y_test_scores >= optimal_threshold).astype(int)

                auc_roc = roc_auc_score(y_test, y_test_scores)
                ap = average_precision_score(y_test, y_test_scores)
                f1 = f1_score(y_test, y_test_pred)
                
                end_time = time.time()
                total_time = end_time - start_time

                print(f"      TEST AUC-ROC: {auc_roc:.4f}")
                print(f"      TEST AP: {ap:.4f}")
                print(f"      TEST F1-Score: {f1:.4f}")
                print(f"      Total Time (s): {total_time:.2f}")

                run_metrics['auc_roc'].append(auc_roc)
                run_metrics['ap'].append(ap)
                run_metrics['f1'].append(f1)
                run_metrics['total_time'].append(total_time)

                if auc_roc > best_auc_for_model:
                    best_auc_for_model = auc_roc
                    best_model_instance = clf
                    best_run_cm = confusion_matrix(y_test, y_test_pred)
            
            if best_model_instance is not None and model_config['model_class'] != eif.iForest:
                model_save_dir = benchmark_results_dir / "saved_models" / dataset_name
                model_save_dir.mkdir(parents=True, exist_ok=True)
                model_filename = f"{model_name.replace(' ', '_')}_best_model.joblib"
                model_save_path = model_save_dir / model_filename
                joblib.dump(best_model_instance, model_save_path)
                print(f"  [INFO] Best {model_name} model saved to: {model_save_path}")
            elif best_model_instance is not None and model_config['model_class'] == eif.iForest:
                print(f"  [INFO] Skipping saving {model_name} model: eif.iForest instances are not directly picklable.")

            if best_run_cm is not None:
                print(f"  [INFO] Confusion Matrix for Best Run (Test AUC: {best_auc_for_model:.4f}):")
                tn, fp, fn, tp = best_run_cm.ravel()
                print(f"    - True Negatives (Normal classified as Normal): {tn}")
                print(f"    - False Positives (Normal classified as Anomaly): {fp}")
                print(f"    - False Negatives (Anomaly classified as Normal): {fn}")
                print(f"    - True Positives (Anomaly classified as Anomaly): {tp}")

            mean_auc_roc = np.mean(run_metrics['auc_roc'])
            std_auc_roc = np.std(run_metrics['auc_roc'])
            
            mean_ap = np.mean(run_metrics['ap'])
            std_ap = np.std(run_metrics['ap'])
            
            mean_f1 = np.mean(run_metrics['f1'])
            std_f1 = np.std(run_metrics['f1'])

            mean_total_time = np.mean(run_metrics['total_time'])
            std_total_time = np.std(run_metrics['total_time'])
            
            result_row = {
                'Dataset': dataset_name,
                'Configuration': model_name,
                'Test AUC': f"{mean_auc_roc:.4f} +/- {std_auc_roc:.4f}",
                'Test AP': f"{mean_ap:.4f} +/- {std_ap:.4f}",
                'Test F1': f"{mean_f1:.4f} +/- {std_f1:.4f}",
                'Total Time (s)': f"{mean_total_time:.2f} +/- {std_total_time:.2f}"
            }
            all_benchmark_results.append(result_row)
            dataset_model_results.append(result_row)
        
        if dataset_model_results:
            df_dataset = pd.DataFrame(dataset_model_results)
            print(f"\n--- Consolidated EIF Results for {dataset_filename} ---")
            print(df_dataset.to_string())
        
    if all_benchmark_results:
        final_df = pd.DataFrame(all_benchmark_results)
        results_file_path = benchmark_results_dir / "eif_benchmark_summary.csv"
        final_df.to_csv(results_file_path, index=False)
        print(f"\n{'='*25} EIF BENCHMARK COMPLETE {'='*25}")
        print(f"All EIF benchmark results saved to {results_file_path}")
        print("\n--- Final EIF Benchmark Summary ---")
        print(final_df.to_string())
    else:
        print("\nNo EIF benchmark results were successfully generated.")


if __name__ == "__main__":
    run_pyod_benchmark() 