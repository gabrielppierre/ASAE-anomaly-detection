"""
Statistical Tests for Model Comparison

This script implements statistical significance tests for comparing
multiple anomaly detection models across different datasets.

Tests implemented:
- Friedman test: Non-parametric test for comparing multiple classifiers
- Nemenyi post-hoc test: Pairwise comparisons after Friedman
- Wilcoxon signed-rank test: Pairwise comparison between two models

Usage:
    python statistical_tests.py --results_dir results/
    python statistical_tests.py --results_csv results/all_results.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
})

# Paleta de cores consistente com o projeto (semi-pastel vibrante)
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', 
          '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

def load_results_from_csv(filepath: str) -> pd.DataFrame:
    """Load results from a summary CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def load_results_from_directory(dirpath: str, metric: str = 'TEST AUC-ROC') -> pd.DataFrame:
    """
    Load results from multiple CSV files in a directory.
    Assumes standard file naming/structure from the benchmark script.
    """
    path = Path(dirpath)
    results = []
    
    for file in path.glob("**/*.csv"):
        try:
            df = pd.read_csv(file)
            # Check if required columns exist
            if 'model' in df.columns and 'dataset' in df.columns and metric in df.columns:
                results.append(df[['dataset', 'model', metric]])
        except:
            continue
            
    if not results:
        print(f"No valid result files found in {dirpath}")
        return pd.DataFrame()
        
    return pd.concat(results, axis=0)

def prepare_comparison_matrix(df: pd.DataFrame, metric: str = 'TEST AUC-ROC') -> pd.DataFrame:
    """
    Prepare a matrix suitable for statistical tests.
    
    Args:
        df: DataFrame with results.
        metric: The metric to compare.
    
    Returns:
        DataFrame with datasets as rows, models as columns.
    """
    # Create a mapping for case-insensitive column lookup
    col_map = {c.lower(): c for c in df.columns}
    
    # Check 1: Is it in Long Format? (Needs pivoting)
    if 'dataset' in col_map and 'model' in col_map:
        # Use the actual column names from the mapping
        dataset_col = col_map['dataset']
        model_col = col_map['model']
        
        # Check if metric column exists
        if metric not in df.columns:
            # Try to find a partial match if exact match fails
            candidates = [c for c in df.columns if metric.lower() in c.lower()]
            if candidates:
                metric = candidates[0]
                print(f"Metric column adjusted to: {metric}")
            else:
                print(f"Warning: Metric '{metric}' not found in columns: {df.columns.tolist()}")
                return pd.DataFrame()
                
        pivot = df.pivot_table(index=dataset_col, columns=model_col, values=metric, aggfunc='mean')
        return pivot
    
    # Check 2: Is it in Wide Format? (Already pivoted, but has 'Dataset' as a column)
    elif 'dataset' in col_map:
        dataset_col = col_map['dataset']
        df = df.set_index(dataset_col)
        
    # Final cleanup: Ensure all data is numeric
    # This fixes the "bad operand type for unary -: 'str'" error
    try:
        df = df.apply(pd.to_numeric, errors='coerce')
        # Drop columns that became all NaN (likely non-numeric columns that weren't the index)
        df = df.dropna(axis=1, how='all')
        # Fill remaining NaNs if any (optional, but safer for stats)
        if df.isnull().values.any():
            print("Warning: NaN values found. Filling with column mean for statistical stability.")
            df = df.fillna(df.mean())
    except Exception as e:
        print(f"Error converting data to numeric: {e}")
        
    return df

def run_friedman_test(df: pd.DataFrame):
    """
    Run Friedman test on the results matrix.
    H0: All algorithms perform equally well.
    """
    print("\n1. FRIEDMAN TEST")
    print("-" * 40)
    
    # Drop rows with NaNs just for the test
    data_matrix = df.dropna().values
    
    if data_matrix.shape[0] < 3:
        print("Not enough datasets (rows) to run Friedman test reliably.")
        return None, None
        
    stat, p_value = stats.friedmanchisquare(*data_matrix.T)
    
    print(f"   χ² statistic: {stat:.4f}")
    print(f"   p-value: {p_value:.4e}")
    
    is_significant = p_value < 0.05
    print(f"   Significant (α=0.05): {'Yes' if is_significant else 'No'}")
    
    return stat, p_value

def run_nemenyi_test(df: pd.DataFrame):
    """
    Run Nemenyi post-hoc test.
    Returns p-values matrix.
    """
    print("\n2. NEMENYI POST-HOC TEST")
    print("-" * 40)
    
    # Convert to matrix format expected by scikit-posthocs
    # rows are blocks (datasets), columns are groups (models)
    # The library expects a melted format or a specific array structure
    
    try:
        # scikit-posthocs expects data where columns are groups (models)
        # and rows are blocks (datasets)
        p_values = sp.posthoc_nemenyi_friedman(df)
        print("   Nemenyi test completed successfully.")
        return p_values
    except Exception as e:
        print(f"   Error running Nemenyi test: {e}")
        return None

def run_wilcoxon_comparisons(df: pd.DataFrame, baseline_model: str = 'ASAE'):
    """
    Run Wilcoxon signed-rank test comparing baseline vs all others.
    """
    print("\n3. WILCOXON SIGNED-RANK TESTS")
    print("-" * 40)
    
    # Find the closest matching column for the baseline
    cols = df.columns.tolist()
    baseline_candidates = [c for c in cols if baseline_model.lower() in c.lower()]
    
    if not baseline_candidates:
        print(f"   Baseline model '{baseline_model}' not found in results.")
        print(f"   Available models: {cols}")
        return None
        
    # Pick the first match (usually the shortest or most exact)
    baseline = baseline_candidates[0] 
    print(f"   Baseline: {baseline}")
    print("   Pairwise comparisons:")
    
    results = []
    
    for model in df.columns:
        if model == baseline:
            continue
            
        try:
            stat, p_val = stats.wilcoxon(df[baseline], df[model])
            
            # Mean difference
            diff = df[baseline] - df[model]
            mean_diff = diff.mean()
            win = (diff > 0).sum()
            loss = (diff < 0).sum()
            tie = (diff == 0).sum()
            
            res = {
                'Model': model,
                'p-value': p_val,
                'Significant': p_val < 0.05,
                'Mean Diff': mean_diff,
                'W-L-T': f"{win}-{loss}-{tie}"
            }
            results.append(res)
            
            sig_mark = "*" if p_val < 0.05 else ""
            print(f"   vs {model[:20]:<20}: p={p_val:.4f} {sig_mark} (Mean Diff: {mean_diff:+.4f})")
            
        except Exception as e:
            print(f"   Could not test vs {model}: {e}")
            
    return pd.DataFrame(results)

def plot_critical_difference(df: pd.DataFrame, output_path: str):
    """
    Plot Critical Difference (CD) diagram.
    This is a simplified visualization of Nemenyi test results.
    """
    try:
        # Calculate average ranks
        ranks = df.rank(axis=1, ascending=False).mean()
        
        # Number of datasets and models
        N = df.shape[0]
        k = df.shape[1]
        
        # Critical difference (approximate for Nemenyi at alpha=0.05)
        # q_alpha values for infinity degrees of freedom
        # This is a simplification; for exact CD diagrams, use orange/biolab libraries
        qa = {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 
            7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268
        }
        q = qa.get(k, 3.3) # Default for k>12
        cd = q * np.sqrt(k * (k + 1) / (6 * N))
        
        plt.figure(figsize=(10, 6))
        
        # Plot ranks
        sorted_ranks = ranks.sort_values()
        y_pos = np.arange(len(sorted_ranks))
        
        plt.barh(y_pos, sorted_ranks.values, color=COLORS[:len(sorted_ranks)], alpha=0.7)
        plt.yticks(y_pos, sorted_ranks.index)
        plt.xlabel('Average Rank (Lower is Better)')
        plt.title(f'Comparison of Average Ranks (CD ≈ {cd:.2f})')
        
        # Add values
        for i, v in enumerate(sorted_ranks.values):
            plt.text(v + 0.1, i, f"{v:.2f}", va='center')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"   CD diagram saved to {output_path}")
        
    except Exception as e:
        print(f"Warning: Could not create CD diagram: {e}")
        import traceback
        traceback.print_exc()

def plot_heatmap(p_values: pd.DataFrame, output_path: str):
    """Plot p-values heatmap."""
    if p_values is None:
        return
        
    plt.figure(figsize=(12, 10))
    
    # Mask diagonal
    mask = np.eye(p_values.shape[0], dtype=bool)
    
    # Create custom colormap: Green for significant (p < 0.05), Red for not
    # Actually, usually low p-value is significant (difference exists)
    # We'll use standard warm/cool map reversed
    
    sns.heatmap(p_values, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                mask=mask, vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'})
    
    plt.title('Nemenyi Post-hoc Test P-values (Green: p < 0.05)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"   Heatmap saved to {output_path}")

def plot_rank_comparison(df: pd.DataFrame, output_path: str):
    """
    Creates a boxplot of the ranks of each model across all datasets.
    """
    # 1. Calculate ranks for each dataset (row)
    # method='min' means ties get the lowest rank (e.g. 1st, 2nd, 2nd, 4th)
    # ascending=False because higher metric (e.g. AUC) is better, so Rank 1 is highest value
    try:
        # Convert to numpy for safer processing
        matrix = df.values
        rows, cols = matrix.shape
        ranks = np.zeros_like(matrix)
        
        for i in range(rows):
            # argsort twice gives ranks
            # We negate matrix[i] because rankdata ranks from low to high
            # and we want high score = rank 1
            ranks[i] = stats.rankdata(-matrix[i])
            
        rank_df = pd.DataFrame(ranks, index=df.index, columns=df.columns)
        
        # Melt for plotting
        melted_ranks = rank_df.melt(var_name='Model', value_name='Rank')
        
        # Sort models by median rank
        order = melted_ranks.groupby('Model')['Rank'].median().sort_values().index
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Model', y='Rank', data=melted_ranks, order=order, palette="viridis")
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Rankings across Datasets (Lower is Better)')
        plt.ylabel('Rank')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"   Rank boxplot saved to {output_path}")
    except Exception as e:
        print(f"   Error plotting rank comparison: {e}")

def create_example_data():
    """Create dummy data for demonstration."""
    models = ['ASAE', 'PyOD-AE', 'IForest', 'LOF', 'OC-SVM']
    datasets = [f'Dataset_{i}' for i in range(1, 11)]
    
    data = {}
    for m in models:
        # Generate random scores with some bias
        base = 0.8 if m == 'ASAE' else 0.75
        scores = np.random.normal(base, 0.05, len(datasets))
        scores = np.clip(scores, 0.5, 1.0)
        data[m] = scores
        
    df = pd.DataFrame(data, index=datasets)
    df.index.name = 'dataset'
    return df

def main():
    parser = argparse.ArgumentParser(description='Statistical Tests for Anomaly Detection')
    parser.add_argument('--results_csv', type=str, help='Path to results summary CSV')
    parser.add_argument('--results_dir', type=str, help='Directory containing result CSVs')
    parser.add_argument('--output_dir', type=str, default='statistical_results',
                       help='Directory to save outputs')
    parser.add_argument('--metric', type=str, default='TEST AUC-ROC',
                       help='Metric column to use for comparison')
    parser.add_argument('--baseline', type=str, default='ASAE',
                       help='Baseline model for Wilcoxon comparisons')
    parser.add_argument('--demo', action='store_true',
                       help='Run with example data for demonstration')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STATISTICAL DATA LOADING")
    print("="*60)
    
    if args.demo:
        print("Running with example data...")
        data = create_example_data()
    elif args.results_csv:
        data = load_results_from_csv(args.results_csv)
        data = prepare_comparison_matrix(data, args.metric)
    elif args.results_dir:
        data = load_results_from_directory(args.results_dir, args.metric)
        data = prepare_comparison_matrix(data, args.metric)
    else:
        print("No data source specified. Use --demo, --results_csv or --results_dir")
        return
    
    if data.empty:
        print("Error: No data to analyze (DataFrame is empty). Check file paths and content.")
        return
        
    print(f"Data Matrix Shape: {data.shape} (Datasets x Models)")
    print(f"Datasets: {data.index.tolist()}")
    print(f"Models: {data.columns.tolist()}")
    
    # Run all tests
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    run_all_tests(data, results_dir, args.baseline)
    
    print("\n" + "="*60)
    print(f"ANALYSIS COMPLETE. Results saved to: {results_dir}")
    print("="*60 + "\n")

def run_all_tests(data, output_dir, baseline):
    # 1. Friedman
    friedman_stat, p_val = run_friedman_test(data)
    
    # Save Friedman results
    with open(output_dir / "friedman_results.txt", "w") as f:
        f.write(f"Friedman Test Results\n")
        f.write(f"Statistic: {friedman_stat}\n")
        f.write(f"P-value: {p_val}\n")
    
    # 2. Nemenyi (only if Friedman is significant usually, but we run anyway for viz)
    if friedman_stat is not None:
        p_values = run_nemenyi_test(data)
        if p_values is not None:
            p_values.to_csv(output_dir / "nemenyi_pvalues.csv")
            plot_heatmap(p_values, str(output_dir / "nemenyi_heatmap.pdf"))
            plot_critical_difference(data, str(output_dir / "cd_diagram.png"))
            
    # 3. Rank Analysis (Robust visualization)
    plot_rank_comparison(data, output_path=str(output_dir / "rank_comparison.png"))

    # 4. Wilcoxon
    wilcox_results = run_wilcoxon_comparisons(data, baseline_model=baseline)
    if wilcox_results is not None:
        wilcox_results.to_csv(output_dir / "wilcoxon_results.csv", index=False)
        
        # Create LaTeX table
        latex = wilcox_results.to_latex(index=False, 
                                      float_format="%.4f",
                                      caption=f"Wilcoxon Signed-Rank Test (Baseline: {baseline})",
                                      label="tab:wilcoxon")
        with open(output_dir / "wilcoxon_table.tex", "w") as f:
            f.write(latex)

if __name__ == "__main__":
    main()