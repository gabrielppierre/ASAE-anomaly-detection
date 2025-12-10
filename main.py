from train import run_for_all_datasets
from data import preprocess_data
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaSSA Anomaly Detection Model Training")
    parser.add_argument('--dataset_path', type=str, default="C:/Users/gabri/Documents/pesquisa_ic/datasets",
                        help='(DEPRECATED) Path to the datasets directory')
    parser.add_argument('--datasets', type=str, nargs='+',
                        required=True,
                        help='List of dataset filenames or full paths to process')
    parser.add_argument('--scaler', type=str, choices=['standard', 'minmax'], default='minmax',
                        help='Type of scaler to use for feature normalization')
    parser.add_argument('--attention', type=str, choices=['adassmax', 'softmax', 'ssmax', 'none', 'sparsemax', 'entmax15', 'performer', 'linformer'], default='adassmax',
                        help='Type of attention mechanism to use. "none" disables attention. "performer" uses Performer attention. "linformer" uses Linformer attention.')
    parser.add_argument('--dynamic_masking', action='store_true', help='Enable dynamic masking rates during training.', default=True)
    parser.add_argument('--inject_noise', action='store_true', help='Inject noise instead of zeros for masking.', default=True)
    parser.add_argument('--temporal_aware', action='store_true', default=False,
                        help='Use temporal-aware data splitting to mitigate concept drift.')
    parser.add_argument('--adaptive_threshold', action='store_true', default=True,
                        help='Use adaptive threshold calculation based on distribution analysis.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save models. If not set, a timestamped directory is created.')
    parser.add_argument('--dataset_fraction', type=float, default=1.0,
                        help='Fraction of the dataset to use (from 0.0 to 1.0). Default is 1.0 (all data).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    print("\n============================== Dataset Selection ==============================")
    if args.datasets:
        print(f"[INFO] Running ONLY on specified datasets:")
        for dataset in args.datasets:
            print(f"  - {dataset}")
    else:
        print("[INFO] No specific datasets specified. Running on all CSV files in the directory.")

    print(f"[INFO] Total datasets to process: {len(args.datasets)}")
    print(f"[INFO] Using random seed: {args.seed}")
    print("================================================================================")

    run_for_all_datasets(
        dataset_paths=args.datasets,
        attention_type=args.attention,
        scaler_type=args.scaler,
        temporal_aware=args.temporal_aware,
        adaptive_threshold=args.adaptive_threshold,
        dynamic_masking=args.dynamic_masking,
        inject_noise=args.inject_noise,
        output_dir=args.output_dir,
        dataset_fraction=args.dataset_fraction,
        random_seed=args.seed
    )
