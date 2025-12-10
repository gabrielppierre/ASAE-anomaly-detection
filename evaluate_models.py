import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import os
import traceback

from data import preprocess_data
from models import AdaptiveAnomalyAutoencoder
from train import compute_adaptive_threshold 
from utils import compute_confidence_interval
import config

def get_model_outputs_composite_score(model, data_loader, device):
    """
    Run the model and return the composite score (reconstruction error + cosine dissimilarity).
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            _, composite_features, _, _, _, _ = model(X_batch)
            
            recon_error_batch = composite_features[:, -2]
            cosine_dissim_batch = composite_features[:, -1]
            composite_score_batch = (recon_error_batch + cosine_dissim_batch).cpu()
            
            all_scores.append(composite_score_batch)
            all_labels.append(y_batch.clone())

    return torch.cat(all_scores), torch.cat(all_labels)


def evaluate_model(model_path, dataset_path, base_seed=42, num_runs=5):
    """
    Evaluate a saved model on a dataset across multiple runs.
    """
    device = config.DEVICE
    run_metrics = []

    print(f"\n--- Evaluating model: {os.path.basename(model_path)} ---")
    print(f"--- Dataset: {os.path.basename(dataset_path)} ---")

    try:
        model_info = torch.load(model_path, map_location=device)
        model_config = model_info['config']
        input_dim = model_info['input_dim']
        
        model = AdaptiveAnomalyAutoencoder(
            input_dim=input_dim,
            attention_type=model_config.get('attention_type', 'adassmax'),
            dynamic_masking=model_config.get('dynamic_masking', True),
            inject_noise=model_config.get('inject_noise', True)
        ).to(device)
        model.load_state_dict(model_info['model_state_dict'])
        model.eval()

    except Exception as e:
        print(f"!!!!!! ERROR loading model file {model_path} !!!!!!")
        print(traceback.format_exc())
        return None

    print(f"  - Preloading {num_runs} data splits...")
    data_splits = []
    for i in range(num_runs):
        run_seed = base_seed + i
        try:
            _, _, X_val, y_val, X_test, y_test, _ = preprocess_data(
                file_path=dataset_path, random_state=run_seed,
                scaler_type='minmax', temporal_aware=True
            )
            data_splits.append((X_val, y_val, X_test, y_test))
        except Exception as e:
            print(f"!!!!!! ERROR preprocessing data for seed {run_seed} !!!!!!")
            print(traceback.format_exc())
            return None
    
    print("  - Preload finished. Starting evaluations...")

    for i, (X_val, y_val, X_test, y_test) in enumerate(data_splits):
        run_seed = base_seed + i
        print(f"  - Run {i+1}/{num_runs} (Seed: {run_seed})...")

        try:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            
            with torch.no_grad():
                val_scores, val_labels = get_model_outputs_composite_score(model, val_loader, device)
                test_scores, test_labels = get_model_outputs_composite_score(model, test_loader, device)

                threshold, _ = compute_adaptive_threshold(
                    val_scores.cpu().numpy(),
                    val_labels.cpu().numpy(),
                    test_scores.cpu().numpy(),
                    strategy='f1',
                    adaptive=False
                )
                print(f"    - Threshold (F1-Optimal): {threshold:.4f}")

                test_preds = (test_scores.cpu().numpy() > threshold).astype(int)

                auc = roc_auc_score(test_labels.cpu().numpy(), test_scores.cpu().numpy())
                ap = average_precision_score(test_labels.cpu().numpy(), test_scores.cpu().numpy())
                f1 = f1_score(test_labels.cpu().numpy(), test_preds)

                run_metrics.append({'AUROC': auc, 'AP': ap, 'F1-Score': f1})

        except Exception as e:
            print(f"!!!!!! ERROR during run {i+1} !!!!!!")
            print(traceback.format_exc())
            continue

    if not run_metrics:
        print(f"--- No successful runs for model {model_path}. Skipping. ---")
        return None

    df_results = pd.DataFrame(run_metrics)
    results_summary = {}
    print(f"\n--- Consolidated results for: {os.path.basename(model_path)} ---")
    for metric in df_results.columns:
        mean_val, std_val, ci = compute_confidence_interval(df_results[metric].values)
        print(f"  - {metric}: {mean_val:.4f} ± {std_val:.4f}")
        results_summary[metric] = {'mean': mean_val, 'std': std_val, 'ci': ci}
    
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script with composite score and F1-optimal threshold")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CSV dataset file to evaluate.')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='List of .pth model files to evaluate.')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for reproducibility.')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs for statistical evaluation.')

    args = parser.parse_args()

    all_model_results = {}
    for model_file in args.model_paths:
        if not os.path.exists(model_file):
            print(f"[WARNING] Model file not found: {model_file}. Skipping.")
            continue
        
        summary = evaluate_model(
            model_path=model_file,
            dataset_path=args.dataset_path,
            base_seed=args.seed,
            num_runs=args.runs
        )
        if summary:
            all_model_results[model_file] = summary

    print("\n=============================== Consolidated Summary ===============================")
    if not all_model_results:
        print("No model was evaluated successfully.")
    else:
        summary_data = []
        for model_path, metrics in all_model_results.items():
            model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            row = {'Modelo': model_name}
            for metric_name, values in metrics.items():
                row[metric_name] = f"{values['mean']:.4f} ± {values['std']:.4f}"
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.set_index('Modelo', inplace=True)
        summary_df.sort_index(inplace=True)
        
        print(summary_df.to_string())

    print("======================================================================================")