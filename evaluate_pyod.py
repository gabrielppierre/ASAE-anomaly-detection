import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from data import preprocess_data # Import directly for 100% consistency

def find_dataset_path(dataset_name, dataset_dirs):
    """Searches for a dataset CSV file in the provided directories."""
    for directory in dataset_dirs:
        # Construct the expected filename
        path = os.path.join(directory, f"{dataset_name}.csv")
        if os.path.exists(path):
            return path
    # A fallback for variations in naming, like hyphens vs underscores
    for directory in dataset_dirs:
        for file in os.listdir(directory):
            if dataset_name.replace('_', '-') in file.replace('_', '-'):
                 return os.path.join(directory, file)
    return None

def get_optimal_f1_threshold(y_true, scores):
    """Find the optimal threshold for F1 score from the precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    # F1 score calculation for each threshold
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

def main():
    # --- IMPORTANT ---
    # Please verify these paths are correct for your system.
    models_dir = '/mnt/hdd/gpcc/baseline/adassa_temporal/codigos/models/'
    dataset_dirs = [
        '/mnt/hdd/gpcc/datasets/cic-ids-2017/tabular/',
        '/mnt/hdd/gpcc/datasets/experiments_cic_ids_18/tabular/'
    ]
    output_csv = 'pyod_metrics_results.csv'
    
    model_paths = glob.glob(os.path.join(models_dir, 'best_model_*.joblib'))
    results = []

    # List of known model identifiers, from longest to shortest to handle overlaps
    known_models = sorted(['AutoEncoder_PyOD', 'VAE_PyOD', 'COPOD', 'ECOD', 'HBOS', 'LOF'], key=len, reverse=True)

    print(f"Found {len(model_paths)} models to evaluate.")

    for model_path in model_paths:
        filename = os.path.basename(model_path)
        print(f"\nProcessing {filename}...")
        
        try:
            # --- Improved Filename Parsing Logic ---
            base_name = filename.replace('best_model_', '').replace('.joblib', '')
            model_name = None
            dataset_name = None

            for model_id in known_models:
                # Check if the base_name ends with the model name, preceded by an underscore
                if base_name.endswith(f'_{model_id}'):
                    model_name = model_id
                    dataset_name = base_name[:-len(f'_{model_id}')]
                    break
            
            if not model_name:
                # Fallback for any models not in the list (e.g., if a new one is added)
                print(f"  - WARNING: Model name in '{filename}' not in known list. Using fallback parsing.")
                parts = base_name.split('_')
                model_name = parts[-1]
                dataset_name = '_'.join(parts[:-1])
            # --- End of Improved Parsing Logic ---

            print(f"  - Model: {model_name}, Dataset: {dataset_name}")

            # Find and preprocess data
            dataset_path = find_dataset_path(dataset_name, dataset_dirs)
            if not dataset_path:
                print(f"  - WARNING: Dataset for '{dataset_name}' not found. Skipping.")
                continue
            
            # Use the imported preprocess_data function to ensure identical processing
            _, _, X_val, y_val, X_test, y_test, _ = preprocess_data(
                file_path=dataset_path,
                temporal_aware=True, # Must be true to match the evaluation logic
                random_state=42
            )

            # Load model
            model = joblib.load(model_path)

            # Get anomaly scores
            if hasattr(model, 'decision_function'):
                val_scores = model.decision_function(X_val)
                test_scores = model.decision_function(X_test)
            else:
                print(f"  - WARNING: Model {model_name} has no decision_function. Skipping.")
                continue

            # Calculate metrics
            auroc = roc_auc_score(y_test, test_scores)
            ap = average_precision_score(y_test, test_scores)
            
            # Determine the best threshold from the validation set for F1 score
            best_threshold = get_optimal_f1_threshold(y_val, val_scores)
            y_pred_test = (test_scores >= best_threshold).astype(int)
            f1 = f1_score(y_test, y_pred_test)

            print(f"  - Metrics: AUROC={auroc:.4f}, AP={ap:.4f}, F1={f1:.4f} (at threshold {best_threshold:.4f})")
            
            results.append({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'AUROC': auroc,
                'AP': ap,
                'F1': f1
            })

        except Exception as e:
            print(f"  - ERROR processing {filename}: {e}")

    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['dataset_name', 'model_name']).reset_index(drop=True)
        print("\n--- Consolidated Results ---")
        print(results_df)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {os.path.abspath(output_csv)}")
    else:
        print("\nNo models were processed successfully.")

if __name__ == '__main__':
    main()
