import os
import sys
os.environ['PYOD_BACKEND'] = 'pytorch'
import time
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import random
import torch
import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.base import clone

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

from data import preprocess_data
from utils import compute_confidence_interval

from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.models.vae import VAE

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

DATASETS_DIR = Path('/mnt/hdd/gpcc/datasets/experiments_cic_ids_18/tabular/')
datasets_to_exclude = {'cic-ids-2017-bot.csv'}

try:
    all_files = [f.name for f in DATASETS_DIR.glob('*.csv') if f.is_file()]
    DATASETS = sorted([f for f in all_files if f not in datasets_to_exclude])
    if not DATASETS:
        print(f"[WARNING] No .csv files found in directory: {DATASETS_DIR}")
    else:
        print(f"[INFO] Datasets found for benchmark: {', '.join(DATASETS)}")
except FileNotFoundError:
    print(f"[ERROR] The specified datasets directory does not exist: {DATASETS_DIR}")
    DATASETS = []

NUM_RUNS = 5

MODELS_TO_BENCHMARK = {
    # 'COPOD': COPOD(),
    # 'ECOD': ECOD(),
    # 'HBOS': HBOS(),
    # 'LOF': LOF(),
    # 'AutoEncoder (PyOD)': AutoEncoder(),
    'VAE (PyOD)': VAE()
}

def set_seed(seed):
    """Seed all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_optimal_threshold(scores_val, labels_val):
    """Select the decision threshold that maximizes F1 on the validation PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(labels_val, scores_val)
    
    if len(thresholds) < len(precisions):
        thresholds = np.append(thresholds, thresholds[-1] + 1e-6)

    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                          out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    if len(f1_scores) > 1:
        f1_scores = f1_scores[:-1]

    if len(f1_scores) == 0:
        return 0.5
        
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate a single model with light preprocessing."""
    model_name = model.__class__.__name__
    print(f"      - Training {model_name}...")

    X_train = np.nan_to_num(np.array(X_train, dtype=np.float32))
    X_val = np.nan_to_num(np.array(X_val, dtype=np.float32))
    X_test = np.nan_to_num(np.array(X_test, dtype=np.float32))
    try:
        selector = VarianceThreshold(threshold=0.0)
        X_train = selector.fit_transform(X_train)
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)
    except ValueError:
        print("      [WARNING] Could not apply VarianceThreshold.")
        pass # Ignora se houver problema (ex: dados vazios)

    if model_name in ['AutoEncoder', 'VAE']:
        model.epochs = 50
        model.batch_size = 128  # Batch size atualizado conforme solicitado
        model.lr = 1e-4
        model.dropout_rate = 0.1
        model.verbose = 0

    start_time = time.perf_counter()
    model.fit(X_train)

    def safe_decision_function(m, X):
        try:
            return m.decision_function(X)
        except Exception as e:
            print(f"      [ERROR] during decision_function: {e}. Returning zeros.")
            return np.zeros(X.shape[0])

    scores_val = safe_decision_function(model, X_val)
    scores_test = safe_decision_function(model, X_test)

    val_auc = roc_auc_score(y_val, scores_val)
    optimal_threshold = find_optimal_threshold(scores_val, y_val)
    test_preds = (scores_test > optimal_threshold).astype(int)
    test_auc = roc_auc_score(y_test, scores_test)
    test_ap = average_precision_score(y_test, scores_test)
    test_f1 = f1_score(y_test, test_preds)
    duration = time.perf_counter() - start_time
    
    print(f"      - {model_name} finished in {duration:.2f}s. "
          f"Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, "
          f"AP: {test_ap:.4f}, F1: {test_f1:.4f}")
    
    return {
        'AUROC': test_auc, 'AP': test_ap, 'F1-Score': test_f1, 'Time (s)': duration
    }, model, val_auc

def execute_run(run_idx, model_name, model_instance, dataset_path, temporal_aware, dataset_fraction):
    """Wrap a single run so it can be parallelized."""
    run_seed = GLOBAL_SEED + run_idx
    set_seed(run_seed)
    
    model_clone = clone(model_instance)
    if hasattr(model_clone, 'random_state'):
        model_clone.set_params(random_state=run_seed)

    print(f"   - Starting Run {run_idx+1}/{NUM_RUNS} for model '{model_name}' (Seed: {run_seed})")
    
    try:
        (
            X_train, y_train, X_val, y_val, X_test, y_test, _
        ) = preprocess_data(
            file_path=str(dataset_path),
            random_state=run_seed,
            temporal_aware=temporal_aware,
            dataset_fraction=dataset_fraction,
            scaler_type='minmax'
        )
        
        if temporal_aware and np.sum(y_train) > 0:
            print(f"      [INFO] Anomalies in training set: {np.sum(y_train)} (expected: 0 for temporal split)")
        
        metrics, trained_model, val_auc = evaluate_model(
            model_clone, X_train, y_train, X_val, y_val, X_test, y_test
        )
        return metrics, trained_model, val_auc

    except Exception as e:
        print(f"   !!!!!! ERROR in run {run_idx+1} for {model_name} on {dataset_path.stem} !!!!!!")
        import traceback
        traceback.print_exc()
        nan_metrics = {'AUROC': np.nan, 'AP': np.nan, 'F1-Score': np.nan, 'Time (s)': np.nan}
        return nan_metrics, None, -1

def run_benchmark(dataset_paths, models, num_runs, temporal_aware, dataset_fraction):
    """Run the full benchmark with parallelized runs."""
    final_results = []
    models_dir = SCRIPT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    for dataset_filename in dataset_paths:
        dataset_path = DATASETS_DIR / dataset_filename
        if not dataset_path.exists():
            print(f"\n[ERROR] Dataset not found at '{dataset_path}'. Skipping.")
            continue
        
        dataset_name = dataset_path.stem
        print(f"\n{'='*25} RUNNING BENCHMARK ON: {dataset_name} {'='*25}")
        
        for model_name, model_instance in models.items():
            print(f"\n--- Model: {model_name} ---")
            
            results = joblib.Parallel(n_jobs=-1, verbose=10)(
                joblib.delayed(execute_run)(
                    i, model_name, model_instance, dataset_path, temporal_aware, dataset_fraction
                ) for i in range(num_runs)
            )
            run_metrics = [res[0] for res in results]
            
            best_run_idx = np.argmax([res[2] for res in results])
            best_run_model = results[best_run_idx][1]
            best_run_val_auc = results[best_run_idx][2]
            
            print(f"   -> Best run for '{model_name}' was Run {best_run_idx+1} with Val AUC: {best_run_val_auc:.4f}")
            
            if best_run_model:
                model_filename = f"best_model_{dataset_name}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
                model_path = models_dir / model_filename
                try:
                    joblib.dump(best_run_model, model_path)
                    print(f"   -> Best model '{model_name}' saved to: {model_path}")
                except Exception as e:
                    print(f"   !!!!!! ERROR saving model '{model_name}': {e} !!!!!!")

            df_run_metrics = pd.DataFrame(run_metrics)
            for metric_name in df_run_metrics.columns:
                mean_val, _, ci_val = compute_confidence_interval(df_run_metrics[metric_name].dropna().values)
                final_results.append({
                    'Dataset': dataset_name, 'Model': model_name, 'Metric': metric_name,
                    'Mean': mean_val, 'CI': ci_val, 'Value': f"{mean_val:.4f} ± {ci_val:.4f}"
                })

    # Imprime e salva os resultados finais
    if final_results:
        df_final = pd.DataFrame(final_results)
        print("\n\n" + "="*30 + " FINAL BENCHMARK RESULTS " + "="*30)
        
        pivot_df = df_final.pivot_table(index=['Dataset', 'Model'], columns='Metric', values='Value', aggfunc='first')
        metric_order = ['AUROC', 'AP', 'F1-Score', 'Time (s)']
        pivot_df = pivot_df[[col for col in metric_order if col in pivot_df.columns]]
        
        print(pivot_df.to_string())
        
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_path = results_dir / f"benchmark_results_{timestamp}.csv"
        pivot_df.to_csv(csv_path)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nNo benchmark results were generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized PyOD Benchmark for Anomaly Detection")
    parser.add_argument('--temporal_aware', action='store_true', default=True,
                        help='Use temporal-aware data splitting (default: True).')
    parser.add_argument('--no-temporal_aware', dest='temporal_aware', action='store_false',
                        help='Disable temporal-aware data splitting.')
    parser.add_argument('--dataset_fraction', type=float, default=1.0,
                        help='Fraction of the dataset to use (from 0.0 to 1.0). Default is 1.0.')
    
    args = parser.parse_args()

    run_benchmark(
        dataset_paths=DATASETS,
        models=MODELS_TO_BENCHMARK,
        num_runs=NUM_RUNS,
        temporal_aware=args.temporal_aware,
        dataset_fraction=args.dataset_fraction
    )