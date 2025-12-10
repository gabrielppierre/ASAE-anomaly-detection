import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

from data import preprocess_data
from models import AdaptiveAnomalyAutoencoder

sns.set_context("paper")
sns.set_style("white")

def load_data_and_extract_attention_weights(dataset_path, adassa_model_path, softmax_model_path):
    """
    Load data, rebuild models from disk, and extract attention weights.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading and preprocessing data...")

    df = pd.read_csv(dataset_path)
    feature_names = df.columns[:-1].tolist()
    
    _, _, X_val, y_val, X_test, y_test, _ = preprocess_data(
        file_path=dataset_path,
        random_state=42,
        scaler_type='minmax',
        temporal_aware=True
    )
    
    X_combined = np.vstack([X_val, X_test])
    y_combined = np.hstack([y_val, y_test])
    
    X_normal = X_combined[y_combined == 0]
    X_anomaly = X_combined[y_combined == 1]
    
    print("Loading models...")
    adassa_model_info = torch.load(adassa_model_path, map_location=device)
    softmax_model_info = torch.load(softmax_model_path, map_location=device)
    
    input_dim = adassa_model_info['input_dim']
    
    adassa_model = AdaptiveAnomalyAutoencoder(
        input_dim=input_dim,
        attention_type='adassmax',
        dynamic_masking=adassa_model_info['config'].get('dynamic_masking', True),
        inject_noise=adassa_model_info['config'].get('inject_noise', True)
    ).to(device)
    adassa_model.load_state_dict(adassa_model_info['model_state_dict'])
    adassa_model.eval()
    
    softmax_model = AdaptiveAnomalyAutoencoder(
        input_dim=input_dim,
        attention_type='softmax',
        dynamic_masking=softmax_model_info['config'].get('dynamic_masking', True),
        inject_noise=softmax_model_info['config'].get('inject_noise', True)
    ).to(device)
    softmax_model.load_state_dict(softmax_model_info['model_state_dict'])
    softmax_model.eval()
    
    print("Extracting attention weights...")
    def extract_attention_weights(model, data):
        model.eval()
        all_weights = []
        batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch_data = torch.FloatTensor(data[i:i+batch_size]).to(device)
                _, _, _, _, _, attn_weights = model(batch_data)
                batch_weights = attn_weights.squeeze(-1).cpu().numpy()
                all_weights.append(batch_weights)
        all_weights = np.vstack(all_weights)
        return np.mean(all_weights, axis=0)
        
    adassa_normal_weights = extract_attention_weights(adassa_model, X_normal)
    adassa_anomaly_weights = extract_attention_weights(adassa_model, X_anomaly)
    softmax_normal_weights = extract_attention_weights(softmax_model, X_normal)
    softmax_anomaly_weights = extract_attention_weights(softmax_model, X_anomaly)

    if len(feature_names) != input_dim:
        print(f"Warning: Feature name count ({len(feature_names)}) does not match input dim ({input_dim}); using generic names (likely due to one-hot encoding).")
        feature_names = [f'Feature_{i}' for i in range(input_dim)]
        
    return (adassa_normal_weights, adassa_anomaly_weights, 
            softmax_normal_weights, softmax_anomaly_weights, 
            feature_names)

def create_clean_dumbbell_plot(adassa_normal, adassa_anomaly, softmax_normal, softmax_anomaly, feature_names):
    """
    Cria um dumbbell plot limpo e alinhado.
    """
    gaps_adassa = np.abs(adassa_anomaly - adassa_normal)
    sorted_indices = np.argsort(gaps_adassa)[::-1]
    
    sorted_features = [feature_names[i] for i in sorted_indices]
    adassa_normal_sorted = adassa_normal[sorted_indices]
    adassa_anomaly_sorted = adassa_anomaly[sorted_indices]
    softmax_normal_sorted = softmax_normal[sorted_indices]
    softmax_anomaly_sorted = softmax_anomaly[sorted_indices]
    
    n_features = len(feature_names)
    y_pos = np.arange(n_features)

    all_values = np.concatenate([
        adassa_normal_sorted, adassa_anomaly_sorted,
        softmax_normal_sorted, softmax_anomaly_sorted
    ])
    x_min = all_values.min()
    x_max = all_values.max()
    padding = (x_max - x_min) * 0.05
    x_limit = (x_min - padding, x_max + padding)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, max(12, n_features * 0.25)), 
                               sharey=True, gridspec_kw={'width_ratios': [4, 2, 4], 'wspace': 0})
    
    ax1, ax_middle, ax2 = axes[0], axes[1], axes[2]

    color_normal = 'steelblue'
    color_anomaly = 'lightcoral'
    color_line = '#cccccc'
    
    for i in range(n_features):
        ax1.plot([adassa_normal_sorted[i], adassa_anomaly_sorted[i]], [y_pos[i], y_pos[i]],
                 color=color_line, linewidth=1, alpha=0.8, zorder=1)
    ax1.scatter(adassa_normal_sorted, y_pos, color=color_normal, s=30, alpha=0.9, zorder=2, label='Normal')
    ax1.scatter(adassa_anomaly_sorted, y_pos, color=color_anomaly, s=30, alpha=0.9, zorder=2, label='Anomaly')
    
    for i in range(n_features):
        ax2.plot([softmax_normal_sorted[i], softmax_anomaly_sorted[i]], [y_pos[i], y_pos[i]],
                 color=color_line, linewidth=1, alpha=0.8, zorder=1)
    ax2.scatter(softmax_normal_sorted, y_pos, color=color_normal, s=30, alpha=0.9, zorder=2)
    ax2.scatter(softmax_anomaly_sorted, y_pos, color=color_anomaly, s=30, alpha=0.9, zorder=2)

    ax_middle.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax_middle.tick_params(left=False, right=False, top=False, bottom=False, labelbottom=False)
    for i, label in enumerate(sorted_features):
        ax_middle.text(0.5, y_pos[i], label, ha='center', va='center', fontsize=9)

    ax1.set_xlabel('Full Model')
    ax2.set_xlabel('Softmax')
    
    ax1.set_xlim(x_limit)
    ax2.set_xlim(x_limit)
    
    ax1.invert_yaxis()
    
    ax1.tick_params(axis='y', length=0, labelleft=False)
    ax2.tick_params(axis='y', length=0)
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main():
    """Entry point to load data, rebuild models, and plot attention weights."""
    dataset_path = "/mnt/hdd/gpcc/datasets/cic-ids-2017/tabular/cic-ids-2017-dos-hulk.csv"
    adassa_model_path = "/mnt/hdd/gpcc/baseline/adassa_temporal/codigos/results/ablation_final_cic-ids-2017-dos-hulk_nao_considerar_a_base/AdaSSA_Full_Model/cic-ids-2017-dos-hulk/best_model_cic-ids-2017-dos-hulk.pth"
    softmax_model_path = "/mnt/hdd/gpcc/baseline/adassa_temporal/codigos/results/ablation_final_cic-ids-2017-dos-hulk_nao_considerar_a_base/Attention_Softmax/cic-ids-2017-dos-hulk/best_model_cic-ids-2017-dos-hulk.pth"
    
    (adassa_normal, adassa_anomaly, 
     softmax_normal, softmax_anomaly, 
     feature_names) = load_data_and_extract_attention_weights(
        dataset_path, adassa_model_path, softmax_model_path
    )
    
    print("Creating visualization...")
    fig = create_clean_dumbbell_plot(
        adassa_normal, adassa_anomaly, 
        softmax_normal, softmax_anomaly, 
        feature_names
    )
    
    output_path = "dumbbell_comparison_adassa_vs_softmax.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Figure saved to: {output_path}")
    
    plt.show()
if __name__ == "__main__":
    main()