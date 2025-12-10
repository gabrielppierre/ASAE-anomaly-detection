#!/usr/bin/env python3
"""
run_full_benchmark.py - Unified benchmark for fair model comparison.

Ensures fair comparison using identical data splits, seeds, and evaluation protocols.
"""

import os
import sys
import time
import json
import argparse
import random
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import joblib

warnings.filterwarnings('ignore')
os.environ['PYOD_BACKEND'] = 'pytorch'

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

from data import preprocess_data
from utils import compute_confidence_interval, count_model_flops, get_pytorch_gpu_memory_peak

from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.models.auto_encoder import AutoEncoder as PyODAutoEncoder
from pyod.models.vae import VAE as PyODVAE
from pyod.models.iforest import IForest

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

from models import AdaptiveAnomalyAutoencoder
import config

GLOBAL_SEED = 42
NUM_RUNS = 1
DATASETS_DIR = Path('datasets/')
DATASETS_TO_EXCLUDE = set()

ASAE_ATTENTION_TYPES = [
    'adassmax', 
    'softmax', 
    'sparsemax', 
    'entmax15', 
    'ssmax', 
    'performer', 
    'linformer', 
]

PYOD_MODELS = {
    'LOF': lambda: LOF(n_neighbors=20, novelty=True),
    'HBOS': lambda: HBOS(n_bins=50),
    'COPOD': lambda: COPOD(),
    'ECOD': lambda: ECOD(),
    'IForest': lambda: IForest(n_estimators=100, random_state=GLOBAL_SEED),
    'AutoEncoder (PyOD)': lambda: PyODAutoEncoder(epoch_num=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, verbose=0),
    'VAE (PyOD)': lambda: PyODVAE(epoch_num=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, verbose=0),
}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def find_optimal_threshold(scores_val, labels_val):
    precisions, recalls, thresholds = precision_recall_curve(labels_val, scores_val)
    if len(thresholds) < len(precisions):
        thresholds = np.append(thresholds, thresholds[-1] + 1e-6)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    
    if len(f1_scores) > 1: f1_scores = f1_scores[:-1]
    if len(f1_scores) == 0: return 0.5
    
    return thresholds[np.argmax(f1_scores)]

def compute_metrics(y_true, scores, threshold=None):
    try:
        if len(np.unique(y_true)) < 2:
            auroc = 0.5
            ap = 0.0
        else:
            auroc = roc_auc_score(y_true, scores)
            ap = average_precision_score(y_true, scores)
    except Exception:
        auroc = 0.5
        ap = 0.0

    if threshold is None:
        threshold = np.percentile(scores, 95)
    
    preds = (scores > threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    
    return {'AUROC': auroc, 'AP': ap, 'F1-Score': f1}

def align_detailed_labels(dataset_path, run_seed, temporal_aware, dataset_fraction):
    """Retrieves detailed labels aligned with X_test by replicating data.py split logic."""
    label_path = dataset_path.parent / f"{dataset_path.stem.replace('_2023', '')}_labels.csv"
    if not label_path.exists():
        label_path = dataset_path.parent / f"{dataset_path.stem}_labels.csv"
        
    if not label_path.exists():
        return None

    try:
        df_labels = pd.read_csv(label_path)
        labels_all = df_labels.iloc[:, 0].values
    except Exception:
        return None

    is_benign = (labels_all == 'Benign')
    indices = np.arange(len(labels_all))
    benign_indices = indices[is_benign]
    attack_indices = indices[~is_benign]
    
    n_train_benign = int(len(benign_indices) * 0.8)
    
    if temporal_aware:
        train_benign_idxs = benign_indices[:n_train_benign]
        pool_benign_idxs = benign_indices[n_train_benign:]
    else:
        train_benign_idxs, pool_benign_idxs = train_test_split(
            benign_indices, train_size=0.8, random_state=run_seed, shuffle=True
        )

    val_test_pool_idxs = np.concatenate([pool_benign_idxs, attack_indices])
    
    _, test_idxs = train_test_split(
        val_test_pool_idxs, test_size=0.5, random_state=run_seed, shuffle=True
    )
    
    return labels_all[test_idxs]


def train_asae_model(model, X_train, y_train, X_val, y_val, device, attention_type, **kwargs):
    from torch.utils.data import DataLoader, TensorDataset
    import copy
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), 
                            batch_size=kwargs.get('batch_size', 128), shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), 
                          batch_size=kwargs.get('batch_size', 128), shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs.get('learning_rate', 1e-4), weight_decay=2e-5)
    
    best_val_auc = 0
    patience = 0
    best_state = None
    num_epochs = kwargs.get('num_epochs', 10)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_b, _ in train_loader:
            X_b = X_b.to(device)
            optimizer.zero_grad()
            rec, comp, _, _, orig, _ = model(X_b)
            loss = torch.nn.functional.mse_loss(rec, orig) + comp[:, -1].mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        scores, labs = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                _, comp, _, _, _, _ = model(X_b)
                scores.append((comp[:, -2] + comp[:, -1]).cpu())
                labs.append(y_b)
        
        try:
            auc = roc_auc_score(torch.cat(labs).numpy(), torch.cat(scores).numpy())
        except:
            auc = 0.5

        avg_loss = total_loss / len(train_loader)
        improved = ""
        if auc > best_val_auc + 1e-4:
            best_val_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            improved = " *"
        else:
            patience += 1
        
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1 or improved:
            print(f"      [Epoch {epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.4f} | Val AUC: {auc:.4f}{improved}")
        
        if patience >= kwargs.get('early_stopping_patience', 5):
            print(f"      Early stopping at epoch {epoch+1}")
            break
            
    if best_state: model.load_state_dict(best_state)
    return model, best_state, best_val_auc

def analyze_per_class(y_test_binary, scores_test, y_test_detailed, threshold):
    """Per-class analysis isolating each attack type against benign samples."""
    unique_classes = np.unique(y_test_detailed)
    
    indices_benign = np.where(y_test_binary == 0)[0]
    benign_label = y_test_detailed[indices_benign[0]] if len(indices_benign) > 0 else 'Benign'
    
    results = []
    for cls in unique_classes:
        if cls == benign_label:
            continue
        mask = (y_test_detailed == benign_label) | (y_test_detailed == cls)
        if not np.any(mask): 
            continue
            
        y_sub_true = y_test_binary[mask]
        scores_sub = scores_test[mask]
        
        metrics = compute_metrics(y_sub_true, scores_sub, threshold)
        metrics['Attack Class'] = cls
        metrics['Count'] = np.sum(y_test_detailed == cls)
        results.append(metrics)
        
    return results

def evaluate_asae(X_train, y_train, X_val, y_val, X_test, y_test, attention_type, seed, device, 
                  y_test_detailed=None, output_dir=None, dataset_name=None, run_idx=0):
    model_name = f"ASAE ({attention_type})"
    print(f"  [{model_name}]", end=" ", flush=True)
    set_all_seeds(seed)
    
    model = AdaptiveAnomalyAutoencoder(
        input_dim=X_train.shape[1], embed_dim=128, p_mask=config.P_MASK,
        dynamic_masking=True, inject_noise=True, attention_type=attention_type,
        alpha_factor=config.ALPHA_FACTOR
    ).to(device)
    
    start_time = time.perf_counter()
    trained_model, best_state, best_val_auc = train_asae_model(
        model, X_train, y_train, X_val, y_val, device, attention_type, 
        num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )
    train_time = time.perf_counter() - start_time
    
    trained_model.eval()
    scores_test = []
    with torch.no_grad():
        for i in range(0, len(X_test), config.BATCH_SIZE):
            X_b = torch.FloatTensor(X_test[i:i+config.BATCH_SIZE]).to(device)
            _, comp, _, _, _, _ = trained_model(X_b)
            scores_test.append((comp[:, -2] + comp[:, -1]).cpu())
    scores_test = torch.cat(scores_test).numpy()
    
    scores_val = []
    with torch.no_grad():
        for i in range(0, len(X_val), config.BATCH_SIZE):
            X_b = torch.FloatTensor(X_val[i:i+config.BATCH_SIZE]).to(device)
            _, comp, _, _, _, _ = trained_model(X_b)
            scores_val.append((comp[:, -2] + comp[:, -1]).cpu())
    scores_val = torch.cat(scores_val).numpy()
    
    threshold = find_optimal_threshold(scores_val, y_val)
    
    metrics = compute_metrics(y_test, scores_test, threshold)
    metrics['Train Time (s)'] = train_time
    
    if output_dir and best_state:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint = {
            'model_state_dict': best_state,
            'attention_type': attention_type,
            'input_dim': X_train.shape[1],
            'embed_dim': 128,
            'best_val_auc': best_val_auc,
            'test_auroc': metrics['AUROC'],
            'test_ap': metrics['AP'],
            'test_f1': metrics['F1-Score'],
            'threshold': threshold,
            'seed': seed,
            'run_idx': run_idx,
            'dataset': dataset_name,
            'config': {
                'p_mask': config.P_MASK,
                'alpha_factor': config.ALPHA_FACTOR,
                'num_epochs': config.NUM_EPOCHS,
                'batch_size': config.BATCH_SIZE,
            }
        }
        ckpt_path = checkpoint_dir / f"{dataset_name}_ASAE_{attention_type}_run{run_idx}.pth"
        if ckpt_path.exists():
            print(f"    [Skip] Checkpoint already exists: {ckpt_path.name}")
        else:
            torch.save(checkpoint, ckpt_path)
            print(f"    [Saved] {ckpt_path.name}")
    
    print(f"AUROC={metrics['AUROC']:.4f} | AP={metrics['AP']:.4f} | F1={metrics['F1-Score']:.4f} | Time={train_time:.1f}s")
    
    detailed_metrics = []
    if y_test_detailed is not None:
        detailed_metrics = analyze_per_class(y_test, scores_test, y_test_detailed, threshold)
    
    return model_name, metrics, detailed_metrics

def evaluate_pyod_model(model_factory, model_name, X_train, y_train, X_val, y_val, X_test, y_test, seed, y_test_detailed=None,
                        output_dir=None, dataset_name=None, run_idx=None):
    print(f"  [{model_name}]", end=" ", flush=True)
    set_all_seeds(seed)
    
    model = model_factory()
    start_time = time.perf_counter()
    model.fit(X_train)
    train_time = time.perf_counter() - start_time
    
    scores_test = model.decision_function(X_test)
    scores_val = model.decision_function(X_val)
    threshold = find_optimal_threshold(scores_val, y_val)
    
    metrics = compute_metrics(y_test, scores_test, threshold)
    metrics['Train Time (s)'] = train_time
    print(f"AUROC={metrics['AUROC']:.4f} | AP={metrics['AP']:.4f} | F1={metrics['F1-Score']:.4f} | Time={train_time:.1f}s")

    if output_dir and dataset_name is not None and run_idx is not None:
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        ckpt_path = ckpt_dir / f"{dataset_name}_{safe_name}_run{run_idx}.joblib"
        if ckpt_path.exists():
            print(f"    [Skip] Checkpoint already exists: {ckpt_path.name}")
        else:
            checkpoint = {
                'model': model,
                'threshold': threshold,
                'seed': seed,
                'run_idx': run_idx,
                'dataset': dataset_name,
                'metrics': {
                    'test_auroc': metrics['AUROC'],
                    'test_ap': metrics['AP'],
                    'test_f1': metrics['F1-Score'],
                    'train_time_s': train_time,
                }
            }
            joblib.dump(checkpoint, ckpt_path)
            print(f"    [Saved] {ckpt_path.name}")
    
    detailed_metrics = []
    if y_test_detailed is not None:
        detailed_metrics = analyze_per_class(y_test, scores_test, y_test_detailed, threshold)
        
    return model_name, metrics, detailed_metrics

def run_full_benchmark(dataset_files, num_runs, temporal_aware, dataset_fraction, output_dir, run_asae, run_pyod, attention_types):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_global_results = []
    all_detailed_results = []
    
    for dataset_path in dataset_files:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_path.name}")
        print(f"{'='*60}")
        
        for run_idx in range(num_runs):
            run_seed = GLOBAL_SEED + run_idx
            print(f"\n--- Run {run_idx + 1}/{num_runs} (Seed: {run_seed}) ---")
            
            try:
                X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_data(
                    str(dataset_path), temporal_aware=temporal_aware, 
                    dataset_fraction=dataset_fraction, random_state=run_seed
                )
                print(f"  Test Set Size (Binary): {len(y_test)} samples")
                
            except Exception as e:
                print(f"  [Error] Preprocessing failed: {e}")
                continue
                
            y_test_detailed = align_detailed_labels(dataset_path, run_seed, temporal_aware, dataset_fraction)
            
            if y_test_detailed is not None and len(y_test_detailed) != len(y_test):
                y_test_detailed = None

            models_to_run = []
            if run_asae:
                for att in attention_types: models_to_run.append(('ASAE', att))
            if run_pyod:
                for name in PYOD_MODELS: models_to_run.append(('PyOD', name))
                
            for model_type, model_key in models_to_run:
                try:
                    if model_type == 'ASAE':
                        m_name, glob_met, det_met = evaluate_asae(
                            X_train, y_train, X_val, y_val, X_test, y_test, 
                            model_key, run_seed, device, y_test_detailed,
                            output_dir=output_dir, dataset_name=dataset_path.stem, run_idx=run_idx
                        )
                    else:
                        m_name, glob_met, det_met = evaluate_pyod_model(
                            PYOD_MODELS[model_key], model_key, 
                            X_train, y_train, X_val, y_val, X_test, y_test, 
                            run_seed, y_test_detailed,
                            output_dir=output_dir, dataset_name=dataset_path.stem, run_idx=run_idx
                        )
                    
                    glob_met['Model'] = m_name
                    glob_met['Dataset'] = dataset_path.stem
                    glob_met['Run'] = run_idx
                    all_global_results.append(glob_met)
                    
                    for d in det_met:
                        d['Model'] = m_name
                        d['Dataset'] = dataset_path.stem
                        d['Run'] = run_idx
                        all_detailed_results.append(d)
                        
                except Exception as e:
                    print(f"  [Error] {model_key}: {e}")
                    import traceback
                    traceback.print_exc()

    if all_global_results:
        pd.DataFrame(all_global_results).to_csv(output_dir / "results_global.csv", index=False)
        print(f"\n[OK] Results saved to {output_dir}/")
        
    if all_detailed_results:
        df_det = pd.DataFrame(all_detailed_results)
        df_det.to_csv(output_dir / "results_per_attack.csv", index=False)
    
    if all_global_results:
        df_global = pd.DataFrame(all_global_results)
        print("\n" + "="*60)
        print("SUMMARY (Mean AUROC per Model)")
        print("="*60)
        summary = df_global.groupby('Model')['AUROC'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        for model, row in summary.iterrows():
            print(f"  {model:<25} {row['mean']:.4f} +/- {row['std']:.4f}")
        
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            n_ckpts = len(list(ckpt_dir.glob("*.pth"))) + len(list(ckpt_dir.glob("*.joblib")))
            print(f"\n[OK] {n_ckpts} checkpoints saved to {ckpt_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--output', type=str, default='results_benchmark')
    parser.add_argument('--run-asae', dest='run_asae', action='store_true', default=True,
                        help='Enable ASAE models (default: on).')
    parser.add_argument('--no-run-asae', dest='run_asae', action='store_false',
                        help='Disable ASAE models to skip retraining them.')
    parser.add_argument('--run-pyod', dest='run_pyod', action='store_true', default=True,
                        help='Enable PyOD baselines (default: on).')
    parser.add_argument('--no-run-pyod', dest='run_pyod', action='store_false',
                        help='Disable PyOD baselines.')
    args = parser.parse_args()
    
    dataset_files = [Path(d) for d in args.datasets]
    
    run_full_benchmark(
        dataset_files=dataset_files,
        num_runs=args.runs,
        temporal_aware=True,
        dataset_fraction=1.0,
        output_dir=args.output,
        run_asae=args.run_asae,
        run_pyod=args.run_pyod,
        attention_types=ASAE_ATTENTION_TYPES
    )