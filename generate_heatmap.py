import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# USER CONFIG
# =============================================================================

# Dataset path (adjust for your environment)
PATH_TO_DATASET = '/mnt/hdd/gpcc/datasets/cic-ids-2017/tabular/cic-ids-2017-dos-hulk.csv'

# Feature and label columns (leave FEATURE_COLUMNS empty to use all but the label)
FEATURE_COLUMNS = [] 
LABEL_COLUMN = 'Label'
NORMAL_CLASS_NAME = 'BENIGN'

# Paths to pretrained models
MODEL_PATHS = {
    'AdaS-SMax': '/mnt/hdd/gpcc/baseline/adassa_temporal/codigos/results/ablation_final_cic-ids-2017-dos-hulk_nao_considerar_a_base/AdaSSA_Full_Model/cic-ids-2017-dos-hulk/best_model_cic-ids-2017-dos-hulk.pth',
    'Softmax': '/mnt/hdd/gpcc/baseline/adassa_temporal/codigos/results/ablation_final_cic-ids-2017-dos-hulk_nao_considerar_a_base/Attention_Softmax/cic-ids-2017-dos-hulk/best_model_cic-ids-2017-dos-hulk.pth',
    'Entmax': '/mnt/hdd/gpcc/baseline/adassa_temporal/codigos/results/ablation_final_cic-ids-2017-dos-hulk_20250705-030423/Attn_Entmax15/cic-ids-2017-dos-hulk/best_model_cic-ids-2017-dos-hulk.pth',
    'Sparsemax': '/mnt/hdd/gpcc/baseline/adassa_temporal/codigos/results/ablation_final_cic-ids-2017-dos-hulk_20250705-030423/Attn_Sparsemax/cic-ids-2017-dos-hulk/best_model_cic-ids-2017-dos-hulk.pth',
    'Base AE': '/mnt/hdd/gpcc/baseline/adassa_temporal/codigos/results/ablation_final_cic-ids-2017-dos-hulk_20250704-142704/Base_AE/cic-ids-2017-dos-hulk/best_model_cic-ids-2017-dos-hulk.pth',
}

# Model definition (must match training architecture)
class Autoencoder(torch.nn.Module):
    def __init__(self, n_features):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_features)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_feature_importance(model, data_loader, device):
    """Compute feature importance via Input x Gradient."""
    model.eval()
    importances = []

    for batch in tqdm(data_loader, desc='Computing importances'):
        inputs = batch.to(device)
        inputs.requires_grad_()
        outputs = model(inputs)
        reconstruction_error = torch.sum((inputs - outputs)**2, dim=1)
        gradients = torch.autograd.grad(torch.sum(reconstruction_error), inputs)[0]
        attribution = inputs.grad.data.abs() 
        importances.append(attribution.cpu().numpy())

    importances = np.concatenate(importances, axis=0)
    return np.mean(importances, axis=0)

def main():
    """Generate comparative heatmaps of feature importance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    df = pd.read_csv(PATH_TO_DATASET)
    if not FEATURE_COLUMNS:
        feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
    else:
        feature_cols = FEATURE_COLUMNS

    df_normal = df[df[LABEL_COLUMN] == NORMAL_CLASS_NAME]
    df_anomaly = df[df[LABEL_COLUMN] != NORMAL_CLASS_NAME]

    X_normal = torch.tensor(df_normal[feature_cols].values, dtype=torch.float32)
    X_anomaly = torch.tensor(df_anomaly[feature_cols].values, dtype=torch.float32)

    normal_loader = torch.utils.data.DataLoader(X_normal, batch_size=256, shuffle=False)
    anomaly_loader = torch.utils.data.DataLoader(X_anomaly, batch_size=256, shuffle=False)

    n_features = len(feature_cols)
    results = {}

    for model_name, model_path in MODEL_PATHS.items():
        print(f'\nProcessing model: {model_name}')
        if not os.path.exists(model_path):
            print(f'WARNING: Model file not found at {model_path}. Skipping.')
            continue

        model = Autoencoder(n_features=n_features).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        imp_normal = get_feature_importance(model, normal_loader, device)
        imp_anomaly = get_feature_importance(model, anomaly_loader, device)

        imp_diff = imp_anomaly - imp_normal
        results[model_name] = imp_diff

    df_heatmap = pd.DataFrame(results, index=feature_cols)

    if 'AdaS-SMax' in df_heatmap.columns:
        sorted_features = df_heatmap['AdaS-SMax'].abs().sort_values(ascending=False).index
        df_heatmap = df_heatmap.loc[sorted_features]

    plt.figure(figsize=(10, 12))
    sns.heatmap(
        df_heatmap,
        cmap='viridis',
        annot=False,
        linewidths=.5
    )
    plt.title('Feature Importance Difference (Anomaly - Normal)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=8)
    plt.tight_layout()

    output_path = '../figura_heatmap_comparativo.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'\nHeatmap saved to: {output_path}')

if __name__ == '__main__':
    if '/mnt/hdd/gpcc/' not in PATH_TO_DATASET and '\\' not in PATH_TO_DATASET:
        print('WARNING: Check if PATH_TO_DATASET is correct for your environment.')
    else:
        main()
