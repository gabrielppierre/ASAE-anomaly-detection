import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import traceback
from scipy.stats import ks_2samp
import os
import pandas as pd
import random

from data import preprocess_data
from models import AdaptiveAnomalyAutoencoder
from utils import compute_confidence_interval, get_pytorch_gpu_memory_peak, hoyer_sparsity, gini_coefficient, count_model_flops, export_runtime_comparison_table
import config

def compute_adaptive_threshold(scores_val, labels_val, scores_test, strategy='f1', adaptive=True):
    """Computes optimal threshold based on validation set using F1 maximization."""
    precisions, recalls, thresholds = precision_recall_curve(labels_val, scores_val)
    if len(thresholds) < len(precisions):
        thresholds = np.append(thresholds, thresholds[-1] + 1e-6)

    if strategy == 'f1':
        f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                              out=np.zeros_like(precisions), where=(precisions + recalls) != 0)[:-1]
        
        if len(f1_scores) == 0:
            return 0.5, {"drift_detected": False, "drift_severity": 0, "adaptation_applied": False}
        
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
    
    else:
        optimal_threshold = np.percentile(scores_val[labels_val == 0], 95)

    return optimal_threshold, {"drift_detected": False, "drift_severity": 0, "adaptation_applied": False}

def get_model_outputs(model, data_loader, device):
    """Extracts composite anomaly scores (MSE + Cosine Dissimilarity) and measures inference time."""
    model.eval()
    all_recon_errors = []
    all_labels = []
    total_loss = 0
    total_inference_time = 0
    num_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            
            inference_start = time.perf_counter()
            reconstructed_x, composite_features, _, _, _, _ = model(X_batch)
            inference_end = time.perf_counter()
            
            total_inference_time += (inference_end - inference_start)
            num_samples += X_batch.size(0)
            
            loss = F.mse_loss(reconstructed_x, X_batch)
            total_loss += loss.item()

            mse_error = composite_features[:, -2].cpu()
            cosine_dissim = composite_features[:, -1].cpu()
            composite_score = mse_error + cosine_dissim
            
            all_recon_errors.append(composite_score)
            all_labels.append(y_batch.clone())

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_inference_time_per_sample = total_inference_time / num_samples if num_samples > 0 else 0
    
    return torch.cat(all_recon_errors), torch.cat(all_labels), avg_loss, avg_inference_time_per_sample

def train_model(file_path, random_seed, attention_type='adassmax', 
                threshold_strategy='f1', scaler_type='standard', 
                temporal_aware=False, adaptive_threshold=True, 
                dynamic_masking=True, inject_noise=True,
                output_dir=None, dataset_fraction=1.0):
    """Trains and evaluates the anomaly detection model for a single run."""
    start_time_total = time.perf_counter()
    device = config.DEVICE
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    try:
        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            scaler
        ) = preprocess_data(file_path, random_state=random_seed, scaler_type=scaler_type, temporal_aware=temporal_aware,
                              dataset_fraction=dataset_fraction)
    except Exception as e:
        print(f"!!!!!! ERROR during data preprocessing for {file_path} !!!!!!")
        traceback.print_exc()
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0, None, np.nan, np.nan, None, None, np.nan

    input_dim = X_train.shape[1]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = AdaptiveAnomalyAutoencoder(
        input_dim=input_dim,
        attention_type=attention_type,
        dynamic_masking=dynamic_masking,
        inject_noise=inject_noise,
        alpha_factor=config.ALPHA_FACTOR
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.LR_SCHEDULER_PATIENCE)
    
    best_val_loss = float('inf')
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    epoch_durations = []

    print(f"  Training for up to {config.NUM_EPOCHS} epochs...")
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.perf_counter()
        model.train()
        total_train_loss = 0
        
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            x_reconstructed, composite_features, _, _, x_original, _ = model(X_batch)
            loss = F.mse_loss(x_reconstructed, x_original)
            
            cosine_dissim = composite_features[:, -1]
            loss += cosine_dissim.mean()
            
            if attention_type == 'adassmax' and model.linear_attn and hasattr(model.linear_attn, 'log_alpha_factor'):
                alpha = torch.exp(model.linear_attn.log_alpha_factor)
                alpha_reg_loss = config.ALPHA_REG_LAMBDA * (alpha ** 2)
                loss += alpha_reg_loss
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        val_recon_errors, val_labels, avg_val_loss, _ = get_model_outputs(model, val_loader, device)
        
        val_auc = roc_auc_score(val_labels, val_recon_errors) if len(np.unique(val_labels)) > 1 else 0.5
        
        temp_threshold, _ = compute_adaptive_threshold(
            val_recon_errors.numpy(), val_labels.numpy(), val_recon_errors.numpy(), 
            strategy='f1', adaptive=False
        )
        val_preds = (val_recon_errors.numpy() > temp_threshold).astype(int)
        val_f1 = f1_score(val_labels.numpy(), val_preds)
        
        scheduler.step(val_auc)
        epoch_duration = time.perf_counter() - epoch_start_time
        epoch_durations.append(epoch_duration)
        
        improved = ""
        if val_auc > best_val_auc + config.MIN_DELTA:
            best_val_auc = val_auc
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            improved = " *"
            if output_dir:
                model_info = {
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_auc': best_val_auc,
                    'input_dim': input_dim,
                    'config': {
                        'attention_type': attention_type,
                        'dynamic_masking': dynamic_masking,
                        'inject_noise': inject_noise,
                    }
                }
                dataset_name = Path(file_path).stem
                model_filename = f"best_model_{dataset_name}.pth"
                torch.save(model_info, os.path.join(output_dir, model_filename))
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or improved:
            print(f"    [Epoch {epoch+1:3d}] Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}{improved}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    avg_epoch_time = np.mean(epoch_durations) if epoch_durations else 0

    if best_model_state is None:
        best_model_state = model.state_dict()
    
    model.load_state_dict(best_model_state)
    model.eval()

    scores_val, labels_val, _, _ = get_model_outputs(model, val_loader, device)
    scores_test, labels_test, _, final_test_inference_time = get_model_outputs(model, test_loader, device)

    final_threshold, drift_info = compute_adaptive_threshold(
        scores_val.numpy(), 
        labels_val.numpy(), 
        scores_test.numpy(),
        strategy=threshold_strategy, 
        adaptive=adaptive_threshold
    )
    
    test_preds = (scores_test.numpy() > final_threshold).astype(int)
    
    final_test_auc = roc_auc_score(labels_test.numpy(), scores_test.numpy()) if len(np.unique(labels_test.numpy())) > 1 else 0.5
    final_test_ap = average_precision_score(labels_test.numpy(), scores_test.numpy()) if len(np.unique(labels_test.numpy())) > 1 else np.mean(labels_test.numpy())
    final_test_f1 = f1_score(labels_test.numpy(), test_preds)
    final_test_precision = precision_score(labels_test.numpy(), test_preds)
    final_test_recall = recall_score(labels_test.numpy(), test_preds)
    
    end_time_total = time.perf_counter()
    overall_wall_time = end_time_total - start_time_total
    
    gpu_memory_peak_mb = get_pytorch_gpu_memory_peak()
    flops = count_model_flops(model, input_dim, device)

    print(f"  Test: AUC={final_test_auc:.4f} | AP={final_test_ap:.4f} | F1={final_test_f1:.4f}")
    
    return final_test_auc, final_test_ap, final_test_f1, final_test_precision, final_test_recall, overall_wall_time, model, best_val_auc, final_threshold, drift_info, best_model_state, avg_epoch_time, final_test_inference_time, gpu_memory_peak_mb, flops

def run_for_all_datasets(dataset_paths, 
                         attention_type='adassmax', scaler_type='standard', 
                         temporal_aware=False, adaptive_threshold=True,
                         dynamic_masking=True, inject_noise=True,
                         output_dir=None, dataset_fraction=1.0, random_seed=42):
    """Runs training and evaluation for a list of dataset paths."""
    all_results = {}
    
    for file_path in dataset_paths:
        file_path = Path(file_path)
        dataset_name = file_path.stem
        
        print(f"\n{'='*20} Processing Dataset: {dataset_name} {'='*20}")

        if output_dir:
            dataset_output_dir = Path(output_dir) / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            dataset_output_dir = None
        
        num_runs = 1
        run_metrics = []

        for i in range(num_runs):
            run_seed = random_seed + i
            print(f"\n--- Starting Run {i+1}/{num_runs} (Seed: {run_seed}) ---")
            
            try:
                test_auc, test_ap, test_f1, test_precision, test_recall, total_time, _, _, _, _, _, time_per_epoch, inference_time, gpu_mem_peak, flops = train_model(
                    file_path=str(file_path),
                    random_seed=run_seed,
                    attention_type=attention_type,
                    scaler_type=scaler_type,
                    temporal_aware=temporal_aware,
                    adaptive_threshold=adaptive_threshold,
                    dynamic_masking=dynamic_masking,
                    inject_noise=inject_noise,
                    output_dir=dataset_output_dir,
                    dataset_fraction=dataset_fraction
                )
                run_metrics.append({
                    'TEST AUC-ROC': test_auc,
                    'TEST AP': test_ap,
                    'TEST F1-Score': test_f1,
                    'Total Time (s)': total_time,
                    'Time per Epoch (s)': time_per_epoch,
                    'Inference Time per Sample (s)': inference_time,
                    'GPU Memory Peak (MB)': gpu_mem_peak if gpu_mem_peak else np.nan,
                    'FLOPs': flops if flops else np.nan,
                })
            except Exception as e:
                print(f"!!!!!! ERROR during training run {i+1} for {dataset_name} !!!!!!")
                traceback.print_exc()
                continue
        
        if not run_metrics:
            print(f"--- No successful runs for dataset {dataset_name}. Skipping. ---")
            continue
        
        df_results = pd.DataFrame(run_metrics)
        
        print(f"\n--- Consolidated Results for: {dataset_name} ---")
        for metric in df_results.columns:
            mean_val, _, ci = compute_confidence_interval(df_results[metric].values)
            print(f"  - {metric}: {mean_val:.4f} +/- {ci:.4f}")
        
        all_results[dataset_name] = df_results
        
        if dataset_output_dir:
            results_csv_path = dataset_output_dir / f"results_{dataset_name}.csv"
            df_results.to_csv(results_csv_path, index=False)
            print(f"  - Results saved to: {results_csv_path}")

    if output_dir and all_results:
        export_runtime_comparison_table(all_results, Path(output_dir))
    
    return all_results