import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import config
from data import preprocess_data
from models import AdaptiveAnomalyAutoencoder


DEFAULT_SEARCH_SPACE = {
    'alpha_reg_lambda': [1e-6, 1e-5, 1e-4, 1e-3],
    'mask_range': [(0.05, 0.15), (0.1, 0.3), (0.15, 0.35), (0.2, 0.4)],
    'noise_std': [0.05, 0.1, 0.15, 0.2],
}

QUICK_SEARCH_SPACE = {
    'alpha_reg_lambda': [1e-5, 1e-4],
    'mask_range': [(0.1, 0.3), (0.2, 0.4)],
    'noise_std': [0.1, 0.2],
}


def train_asae_with_hyperparams(X_train, y_train, X_val, y_val, X_test, y_test,
                                alpha_reg_lambda: float,
                                mask_range: Tuple[float, float],
                                noise_std: float,
                                seed: int = 42) -> Dict:
    """
    Train ASAE model with specified hyperparameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        alpha_reg_lambda: Regularization strength for alpha
        mask_range: Tuple of (min, max) masking probability
        noise_std: Standard deviation for noise injection
        seed: Random seed
    
    Returns:
        Dictionary with training results
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = config.DEVICE
    input_dim = X_train.shape[1]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = AdaptiveAnomalyAutoencoder(
        input_dim=input_dim,
        attention_type='adassmax',
        dynamic_masking=True,
        mask_range=mask_range,
        inject_noise=True,
        noise_std=noise_std,
        alpha_factor=config.ALPHA_FACTOR
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.LR_SCHEDULER_PATIENCE)
    
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    train_start = time.perf_counter()
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            
            x_reconstructed, composite_feature, scores, mask, x_original, attn_weights = model(X_batch)
            
            mse_loss = nn.MSELoss()(x_reconstructed, x_original)
            
            alpha_reg = 0
            if hasattr(model.linear_attn, 'attention') and hasattr(model.linear_attn.attention, 'alpha_param'):
                alpha_val = 1.0 + torch.nn.functional.softplus(model.linear_attn.attention.alpha_param)
                alpha_reg = alpha_reg_lambda * (alpha_val - 1.0) ** 2
            
            loss = mse_loss + alpha_reg
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        model.eval()
        val_scores = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                x_reconstructed, composite_feature, scores, mask, x_original, attn_weights = model(X_batch)
                
                batch_scores = torch.mean((x_reconstructed - X_batch) ** 2, dim=1)
                val_scores.extend(batch_scores.cpu().numpy())
                val_labels.extend(y_batch.numpy())
        
        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)
        
        val_auc = roc_auc_score(val_labels, val_scores)
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc + config.MIN_DELTA:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                break
    
    train_time = time.perf_counter() - train_start
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_scores = []
    test_labels = []
    
    inference_start = time.perf_counter()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            x_reconstructed, composite_feature, scores, mask, x_original, attn_weights = model(X_batch)
            
            batch_scores = torch.mean((x_reconstructed - X_batch) ** 2, dim=1)
            test_scores.extend(batch_scores.cpu().numpy())
            test_labels.extend(y_batch.numpy())
    
    inference_time = time.perf_counter() - inference_start
    
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
    
    test_auc = roc_auc_score(test_labels, test_scores)
    test_ap = average_precision_score(test_labels, test_scores)
    
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(test_labels, test_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    test_f1 = f1_scores[best_f1_idx]
    
    return {
        'test_auc': test_auc,
        'test_ap': test_ap,
        'test_f1': test_f1,
        'val_auc': best_val_auc,
        'train_time': train_time,
        'inference_time': inference_time
    }


def run_single_experiment(dataset_path: str, 
                          alpha_reg_lambda: float,
                          mask_range: Tuple[float, float],
                          noise_std: float,
                          seed: int = 42,
                          output_dir: str = None) -> Dict:
    """
    Run a single training experiment with specified hyperparameters.
    
    Args:
        dataset_path: Path to the dataset CSV file.
        alpha_reg_lambda: Regularization strength for alpha.
        mask_range: Tuple of (min, max) masking probability.
        noise_std: Standard deviation for noise injection.
        seed: Random seed for reproducibility.
        output_dir: Directory to save model checkpoints.
    
    Returns:
        Dictionary with experiment results.
    """
    try:
        data = preprocess_data(
            file_path=dataset_path,
            random_state=seed,
            scaler_type='standard',
            temporal_aware=True,
            dataset_fraction=1.0
        )
        
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = data
        
        results = train_asae_with_hyperparams(
            X_train, y_train, X_val, y_val, X_test, y_test,
            alpha_reg_lambda=alpha_reg_lambda,
            mask_range=mask_range,
            noise_std=noise_std,
            seed=seed
        )
        
        return {
            'alpha_reg_lambda': alpha_reg_lambda,
            'mask_range': str(mask_range),
            'noise_std': noise_std,
            'test_auc': results['test_auc'],
            'test_ap': results['test_ap'],
            'test_f1': results['test_f1'],
            'val_auc': results['val_auc'],
            'train_time': results['train_time'],
            'inference_time': results['inference_time'],
            'success': True
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'alpha_reg_lambda': alpha_reg_lambda,
            'mask_range': str(mask_range),
            'noise_std': noise_std,
            'test_auc': np.nan,
            'test_ap': np.nan,
            'test_f1': np.nan,
            'val_auc': np.nan,
            'train_time': np.nan,
            'inference_time': np.nan,
            'success': False,
            'error': str(e)
        }


def run_grid_search(dataset_path: str,
                    search_space: Dict,
                    n_seeds: int = 3,
                    output_dir: str = './sensitivity_results') -> pd.DataFrame:
    """
    Run grid search over hyperparameter combinations.
    
    Args:
        dataset_path: Path to the dataset CSV file.
        search_space: Dictionary mapping hyperparameter names to lists of values.
        n_seeds: Number of random seeds for each configuration.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame with all experiment results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    combinations = list(itertools.product(*param_values))
    
    total_experiments = len(combinations) * n_seeds
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Search space: {len(combinations)} configurations × {n_seeds} seeds = {total_experiments} experiments")
    print(f"Parameters: {param_names}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            for seed in range(42, 42 + n_seeds):
                pbar.set_postfix({k: str(v)[:10] for k, v in params.items()})
                
                mask_range_val = params['mask_range']
                if isinstance(mask_range_val, str):
                    mask_range_val = eval(mask_range_val)
                
                result = run_single_experiment(
                    dataset_path=dataset_path,
                    alpha_reg_lambda=params['alpha_reg_lambda'],
                    mask_range=mask_range_val,
                    noise_std=params['noise_std'],
                    seed=seed
                )
                result['seed'] = seed
                all_results.append(result)
                
                pbar.update(1)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    df = pd.DataFrame(all_results)
    
    df.to_csv(output_path / 'sensitivity_raw_results.csv', index=False)
    
    agg_df = df.groupby(['alpha_reg_lambda', 'mask_range', 'noise_std']).agg({
        'test_auc': ['mean', 'std'],
        'test_ap': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'train_time': 'mean'
    }).round(4)
    
    agg_df.to_csv(output_path / 'sensitivity_aggregated.csv')
    
    print(f"\nResults saved to: {output_path}")
    
    return df


def analyze_single_parameter(df: pd.DataFrame, param_name: str, 
                            metric: str = 'test_auc') -> pd.DataFrame:
    """
    Analyze the effect of a single parameter on performance.
    
    Args:
        df: DataFrame with experiment results.
        param_name: Name of the parameter to analyze.
        metric: Metric to use for analysis.
    
    Returns:
        DataFrame with analysis results.
    """
    analysis = df.groupby(param_name).agg({
        metric: ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    
    analysis.columns = ['mean', 'std', 'min', 'max', 'n_experiments']
    
    return analysis


def plot_sensitivity_heatmap(df: pd.DataFrame, 
                             param_x: str, param_y: str,
                             metric: str = 'test_auc',
                             output_path: str = None) -> None:
    """
    Create a heatmap showing sensitivity to two parameters.
    
    Args:
        df: DataFrame with experiment results.
        param_x: Parameter for x-axis.
        param_y: Parameter for y-axis.
        metric: Metric to visualize.
        output_path: Path to save the figure.
    """
    pivot = df.groupby([param_x, param_y])[metric].mean().unstack()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                center=pivot.values.mean())
    
    plt.title(f'Sensitivity Analysis: {metric}')
    plt.xlabel(param_y)
    plt.ylabel(param_x)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
    
    plt.close()


def plot_parameter_effect(df: pd.DataFrame, param_name: str,
                          metric: str = 'test_auc',
                          output_path: str = None) -> None:
    """
    Create a line plot showing the effect of a parameter on performance.
    
    Args:
        df: DataFrame with experiment results.
        param_name: Parameter to analyze.
        metric: Metric to visualize.
        output_path: Path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    
    grouped = df.groupby(param_name)[metric].agg(['mean', 'std'])
    
    x = range(len(grouped))
    labels = grouped.index.tolist()
    
    plt.errorbar(x, grouped['mean'], yerr=grouped['std'], 
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    
    plt.xticks(x, labels, rotation=45)
    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.title(f'Effect of {param_name} on {metric}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.close()


def generate_all_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate all sensitivity analysis plots.
    
    Args:
        df: DataFrame with experiment results.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics = ['test_auc', 'test_ap', 'test_f1']
    params = ['alpha_reg_lambda', 'mask_range', 'noise_std']
    
    print("\nGenerating plots...")
    
    for param in params:
        for metric in metrics:
            plot_parameter_effect(
                df, param, metric,
                output_path=str(output_path / f'effect_{param}_{metric}.png')
            )
    
    param_pairs = [
        ('alpha_reg_lambda', 'noise_std'),
        ('alpha_reg_lambda', 'mask_range'),
        ('mask_range', 'noise_std')
    ]
    
    for param_x, param_y in param_pairs:
        for metric in metrics:
            plot_sensitivity_heatmap(
                df, param_x, param_y, metric,
                output_path=str(output_path / f'heatmap_{param_x}_{param_y}_{metric}.png')
            )
    
    print(f"All plots saved to: {output_path}")


def generate_latex_table(df: pd.DataFrame) -> str:
    """
    Generate LaTeX table with sensitivity analysis results.
    
    Args:
        df: DataFrame with experiment results.
    
    Returns:
        LaTeX code as string.
    """
    best_idx = df['test_auc'].idxmax()
    best_config = df.loc[best_idx]
    
    latex = []
    latex.append("% Hyperparameter Sensitivity Analysis Results")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Hyperparameter Sensitivity Analysis}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Parameter} & \\textbf{Value} & \\textbf{AUC-ROC} & \\textbf{F1-Score} \\\\")
    latex.append("\\midrule")
    
    for param in ['alpha_reg_lambda', 'mask_range', 'noise_std']:
        grouped = df.groupby(param).agg({
            'test_auc': 'mean',
            'test_f1': 'mean'
        }).round(4)
        
        latex.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{param}}}}} \\\\")
        for idx, row in grouped.iterrows():
            latex.append(f"  & {idx} & {row['test_auc']:.4f} & {row['test_f1']:.4f} \\\\")
    
    latex.append("\\midrule")
    latex.append(f"\\textbf{{Best Config}} & - & \\textbf{{{best_config['test_auc']:.4f}}} & \\textbf{{{best_config['test_f1']:.4f}}} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sensitivity analysis for ASAE')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='./sensitivity_results',
                       help='Directory to save results')
    parser.add_argument('--n_seeds', type=int, default=3,
                       help='Number of random seeds per configuration')
    parser.add_argument('--quick', action='store_true',
                       help='Use reduced search space for faster testing')
    parser.add_argument('--analyze_only', type=str, default=None,
                       help='Path to existing results CSV to analyze (skip training)')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.analyze_only:
        print(f"Loading existing results from: {args.analyze_only}")
        df = pd.read_csv(args.analyze_only)
    else:
        search_space = QUICK_SEARCH_SPACE if args.quick else DEFAULT_SEARCH_SPACE
        
        df = run_grid_search(
            dataset_path=args.dataset,
            search_space=search_space,
            n_seeds=args.n_seeds,
            output_dir=args.output_dir
        )
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    for param in ['alpha_reg_lambda', 'mask_range', 'noise_std']:
        print(f"\nEffect of {param}:")
        analysis = analyze_single_parameter(df, param)
        print(analysis)
    
    if df['test_auc'].notna().any():
        best_idx = df['test_auc'].idxmax()
        best_config = df.loc[best_idx]
        
        print("\n" + "="*60)
        print("BEST CONFIGURATION")
        print("="*60)
        print(f"  α regularization (λ_α): {best_config['alpha_reg_lambda']}")
        print(f"  Mask range: {best_config['mask_range']}")
        print(f"  Noise std: {best_config['noise_std']}")
        print(f"  Test AUC-ROC: {best_config['test_auc']:.4f}")
        print(f"  Test AP: {best_config['test_ap']:.4f}")
        print(f"  Test F1: {best_config['test_f1']:.4f}")
    else:
        print("\n" + "="*60)
        print("WARNING: All experiments failed!")
        print("="*60)
        print("Check error messages above for details.")
        return
    
    generate_all_plots(df, args.output_dir)
    
    latex_code = generate_latex_table(df)
    latex_path = output_path / 'sensitivity_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    print(f"\nLaTeX table saved to: {latex_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
