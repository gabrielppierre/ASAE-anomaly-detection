import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# Paste evaluation results here
results_text = """
Modelo                                  AUROC               AP         F1-Score
AdaSSA_Full_Model       0.9614 ± 0.0002  0.7154 ± 0.0250  0.8528 ± 0.0084
AdaSSA_dynamic_masking  0.9510 ± 0.0003  0.6592 ± 0.0114  0.8372 ± 0.0026
AdaSSA_inject_noise     0.9596 ± 0.0002  0.7013 ± 0.0111  0.8475 ± 0.0024
Attention_None          0.9537 ± 0.0002  0.7013 ± 0.0111  0.8454 ± 0.0024
Attention_Softmax       0.9541 ± 0.0002  0.6995 ± 0.0112  0.8326 ± 0.0025
Attn_Entmax15           0.9389 ± 0.0003  0.6313 ± 0.0114  0.8089 ± 0.0026
Attn_Sparsemax          0.9468 ± 0.0003  0.6599 ± 0.0112  0.8205 ± 0.0027
Base_AE                 0.9488 ± 0.0002  0.6803 ± 0.0113  0.8348 ± 0.0025
"""

# Name mapping and plot order
model_name_map = {
    'AdaSSA_Full_Model': 'Full Model',
    'AdaSSA_dynamic_masking': 'w/o Dyn. Masking',
    'AdaSSA_inject_noise': 'w/o Noise',
    'Attention_None': 'w/o Attention',
    'Attention_Softmax': 'Attn: Softmax',
    'Attn_Entmax15': 'Attn: Entmax1.5',
    'Attn_Sparsemax': 'Attn: Sparsemax',
    'Base_AE': 'Base AE'
}

# Model order in the plot (bottom to top)
plot_order = [
    'Full Model',
    'w/o Dyn. Masking',
    'w/o Noise',
    'Attn: Softmax',
    'Attn: Entmax1.5',
    'Attn: Sparsemax',
    'w/o Attention',
    'Base AE'
]

def parse_results(text):
    """Convert the raw results text into a pandas DataFrame."""
    data = io.StringIO(text.strip())
    df = pd.read_csv(data, sep='|', header=None)

    regex = r'(\S+)\s+([\d.]+\s±\s[\d.]+)\s+([\d.]+\s±\s[\d.]+)\s+([\d.]+\s±\s[\d.]+)'
    
    results = df[1:].iloc[:, 0].str.extract(regex)
    results.columns = ['Modelo', 'AUROC', 'AP', 'F1-Score']
    results.set_index('Modelo', inplace=True)

    for col in ['AUROC', 'AP', 'F1-Score']:
        split_cols = results[col].str.split('\s±\s', expand=True)
        results[f'{col}_mean'] = pd.to_numeric(split_cols[0])
        results[f'{col}_err'] = pd.to_numeric(split_cols[1])
    
    return results.drop(columns=['AUROC', 'AP', 'F1-Score'])

def plot_ablation_study(df, metric, ax, full_model_score):
    """Create a single bar plot for the ablation study."""
    colors = df.index.map(color_map)
    
    ax.barh(df.index, df[f'{metric}_mean'], xerr=df[f'{metric}_err'], 
            align='center', color=colors, capsize=5)

    ax.axvline(x=full_model_score, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=f'Full Model Score: {full_model_score:.4f}')

    for i, model_name in enumerate(df.index):
        mean = df.loc[model_name, f'{metric}_mean']
        err = df.loc[model_name, f'{metric}_err']

        delta = ((mean / full_model_score) - 1) * 100
        sign = '+' if delta >= 0 else ''
        annotation = f'{mean:.4f}*\n(Δ {sign}{delta:.1f}%)'
        offset = 0.001
        ax.text(mean + err + offset, i, annotation, 
                va='center', ha='left', fontsize=10, color='darkred', fontweight='bold')

    ax.invert_yaxis() # Modelo base no topo, completo na base
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)

    min_val = df[f'{metric}_mean'].min() - 0.03
    max_val = df[f'{metric}_mean'].max() + 0.05
    ax.set_xlim([min_val, max_val])


color_map = {
    'Full Model': '#485460',        # Dark Slate Gray/Blue
    'w/o Dyn. Masking': '#c39bd3',   # Medium Lavender
    'w/o Noise': '#f1948a',         # Light Coral
    'w/o Attention': '#f0b27a',     # Medium Peach
    'Attn: Softmax': '#82e0aa',      # Medium Mint Green
    'Attn: Entmax1.5': '#76d7c4',    # Medium Teal
    'Attn: Sparsemax': '#73c6b6',    # Medium Aqua/Teal
    'Base AE': '#85c1e9'            # Medium Steel Blue
}

if __name__ == "__main__":
    results_df = parse_results(results_text)
    results_df.index = results_df.index.map(model_name_map)
    results_df = results_df.reindex(plot_order)

    full_model_ap = results_df.loc['Full Model']['AP_mean']
    full_model_f1 = results_df.loc['Full Model']['F1-Score_mean']

    plt.style.use('seaborn-v0_8-whitegrid')

    fig_ap, ax_ap = plt.subplots(figsize=(8, 6))
    plot_ablation_study(results_df, 'AP', ax_ap, full_model_ap)
    output_ap = 'ablation_study_AP.pdf'
    fig_ap.tight_layout()
    fig_ap.savefig(output_ap, format='pdf', dpi=300, bbox_inches='tight')
    print(f"AP plot saved to '{output_ap}'.")

    fig_f1, ax_f1 = plt.subplots(figsize=(8, 6))
    plot_ablation_study(results_df, 'F1-Score', ax_f1, full_model_f1)
    output_f1 = 'ablation_study_F1.pdf'
    fig_f1.tight_layout()
    fig_f1.savefig(output_f1, format='pdf', dpi=300, bbox_inches='tight')
    print(f"F1-Score plot saved to '{output_f1}'.")

    # Optional: display the plots
    # plt.show()
