import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

results_text = ""

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

    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)

    min_val = df[f'{metric}_mean'].min() - 0.03
    max_val = df[f'{metric}_mean'].max() + 0.05
    ax.set_xlim([min_val, max_val])


color_map = {
    'Full Model': '#485460',
    'w/o Dyn. Masking': '#c39bd3',
    'w/o Noise': '#f1948a',
    'w/o Attention': '#f0b27a',
    'Attn: Softmax': '#82e0aa',
    'Attn: Entmax1.5': '#76d7c4',
    'Attn: Sparsemax': '#73c6b6',
    'Base AE': '#85c1e9'
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