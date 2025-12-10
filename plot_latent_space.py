import torch
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import traceback
from sklearn.metrics import precision_recall_curve

from models import AdaptiveAnomalyAutoencoder
from data import preprocess_data

def get_latent_space_and_scores(model, data):
    """
    Run the model to extract latent representations and anomaly scores.
    """
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        data_tensor = torch.FloatTensor(data).to(device)
        _, composite_features, _, _, _, _ = model(data_tensor)

        latent_rep = composite_features[:, :model.embedding.out_features].cpu().numpy()
        anomaly_scores = composite_features[:, -2].cpu().numpy()

    if np.any(np.isnan(latent_rep)) or np.any(np.isinf(latent_rep)):
        print("\n*** WARNING: NaN or Inf detected in latent space. Replacing with zeros. ***\n")
        latent_rep = np.nan_to_num(latent_rep, nan=0.0, posinf=0.0, neginf=0.0)

    return latent_rep, anomaly_scores

def compute_adaptive_threshold(scores_val, labels_val):
    """
    Compute the threshold that maximizes validation F1.
    """
    precisions, recalls, thresholds = precision_recall_curve(labels_val, scores_val)
    if len(thresholds) < len(precisions):
        thresholds = np.append(thresholds, thresholds[-1] + 1e-6)

    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                          out=np.zeros_like(precisions), where=(precisions + recalls) != 0)[:-1]
    
    if len(f1_scores) == 0:
        return np.percentile(scores_val, 95)  # Fallback
    
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    
    return optimal_threshold

def get_model_config_from_path(model_path):
    """Infer a lightweight config from the model path."""
    path_lower = model_path.lower()
    
    config = {'attention_type': 'adassmax', 'dynamic_masking': True, 'inject_noise': True}
    if 'base_ae' in path_lower:
        return {'attention_type': 'none', 'dynamic_masking': False, 'inject_noise': False}
    if 'adassa_full_model' in path_lower:
        return config

    if 'w_o_dynamic_masking' in path_lower: config['dynamic_masking'] = False
    if 'w_o_inject_noise' in path_lower: config['inject_noise'] = False
    if 'attention_none' in path_lower: config['attention_type'] = 'none'
    if 'attention_softmax' in path_lower: config['attention_type'] = 'softmax'
    if 'attn_entmax15' in path_lower: config['attention_type'] = 'entmax15'
    if 'attn_sparsemax' in path_lower: config['attention_type'] = 'sparsemax'

    print(f"Inferred config for {os.path.basename(model_path)}: {config}")
    return config

def plot_latent_space(latent_space, true_labels, anomaly_scores, threshold, model_name, output_dir):
    """Apply UMAP and plot a 2D scatter colored by detection outcomes."""
    print(f"[{model_name}] Applying UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    latent_2d = reducer.fit_transform(latent_space)

    predicted_labels = (anomaly_scores >= threshold).astype(int)

    df_plot = pd.DataFrame({
        'dim1': latent_2d[:, 0], 'dim2': latent_2d[:, 1],
        'true_label': true_labels, 'predicted_label': predicted_labels
    })

    def assign_category(row):
        if row['true_label'] == 0 and row['predicted_label'] == 0:
            return 'True Negative'
        elif row['true_label'] == 1 and row['predicted_label'] == 0:
            return 'False Negative'
        elif row['true_label'] == 1 and row['predicted_label'] == 1:
            return 'True Positive'
        else: # true_label == 0 and predicted_label == 1
            return 'False Positive'

    df_plot['category'] = df_plot.apply(assign_category, axis=1)
    df_plot = df_plot[df_plot['category'] != 'False Positive']

    palette = {
        'True Negative': 'grey',
        'False Negative': 'orange',
        'True Positive': 'blue',
    }
    markers = {
        'True Negative': 'o',
        'False Negative': 's',
        'True Positive': 'X',
    }
    hue_order = list(palette.keys())

    print(f"[{model_name}] Generating plot...")
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df_plot, x='dim1', y='dim2',
        hue='category', style='category',
        palette=palette, markers=markers, hue_order=hue_order, style_order=hue_order,
        s=70, alpha=0.8, legend=False
    )
    
    plt.title('')
    plt.xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')

    output_filename = f"latent_space_{model_name}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model latent space.')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Paths to model files (.pth).')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save plots.')
    parser.add_argument('--data_percentage', type=float, default=100.0, help='Percentage of data to use.')
    parser.add_argument('--plot_sample_size', type=int, default=50000, help='Max number of points to plot.')
    args = parser.parse_args()

    print("--- STARTING LATENT SPACE VISUALIZATION ---")

    _, _, X_val, y_val, X_test, y_test, _ = preprocess_data(
        file_path=args.data_path, random_state=42, scaler_type='minmax',
        temporal_aware=True, dataset_fraction=args.data_percentage / 100.0
    )

    n_features = X_test.shape[1]
    os.makedirs(args.output_dir, exist_ok=True)

    for model_path in args.model_paths:
        print(f"\n--- Processing: {os.path.basename(model_path)} ---")
        try:
            config_params = get_model_config_from_path(model_path)
            model = AdaptiveAnomalyAutoencoder(
                input_dim=n_features, embed_dim=128,
                **config_params
            )
            
            model_info = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_info['model_state_dict'])
            print("Model loaded successfully.")

            _, val_anomaly_scores = get_latent_space_and_scores(model, X_val)
            threshold = compute_adaptive_threshold(val_anomaly_scores, y_val)
            print(f"Optimal threshold: {threshold:.6f}")

            test_latent_rep, test_anomaly_scores = get_latent_space_and_scores(model, X_test)

            if len(y_test) > args.plot_sample_size:
                indices = np.random.choice(len(y_test), args.plot_sample_size, replace=False)
                plot_latent_rep, plot_y, plot_scores = test_latent_rep[indices], y_test[indices], test_anomaly_scores[indices]
            else:
                plot_latent_rep, plot_y, plot_scores = test_latent_rep, y_test, test_anomaly_scores

            model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            plot_latent_space(plot_latent_rep, plot_y, plot_scores, threshold, model_name, args.output_dir)

        except Exception as e:
                print(f"Failed to process {model_path}. Error: {e}")
            traceback.print_exc()
            
            print("\n--- PROCESS FINISHED ---")