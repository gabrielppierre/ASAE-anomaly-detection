import numpy as np
import torch
import scipy.stats as st


try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
except pynvml.NVMLError as error:
    HAS_NVML = False 
    print(f"Warning: NVML found but failed to initialize: {error}. GPU metrics disabled.")

def get_gpu_power():
    if HAS_NVML and torch.cuda.is_available():
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power_mW / 1000.0
    return None

def get_gpu_memory_usage():
    cuda_available = torch.cuda.is_available()
    
    if HAS_NVML and cuda_available:
        try:
            device_index = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = mem_info.used / (1024 * 1024.0)
            return memory_used_mb
        except pynvml.NVMLError as err:
            return None
        except Exception as err:
            return None
    else:
        return None

def get_pytorch_gpu_memory_peak():
    if torch.cuda.is_available():
        try:
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / (1024 * 1024.0)
            return peak_mb
        except Exception as err:
            print(f"Unexpected Error getting PyTorch GPU memory peak: {err}")
            return None
    else:
        return None

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return np.mean(data) if n > 0 else np.nan, np.nan, np.nan 
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    if std_dev < 1e-9:
        return mean, 0.0, 0.0 
    std_err = std_dev / np.sqrt(n)
    h = std_err * st.t.ppf((1 + confidence) / 2., n - 1)
    return mean, std_dev, h

def hoyer_sparsity(weights):
    l2_norm = torch.norm(weights, p=2)
    l1_norm = torch.norm(weights, p=1)
    return l2_norm / l1_norm if l1_norm > 0 else 0

def gini_coefficient(weights):
    sorted_weights, _ = torch.sort(weights.flatten())
    n = sorted_weights.size(0)
    cumsum = torch.cumsum(sorted_weights, dim=0)
    return (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n if cumsum[-1] > 0 else 0


def count_model_flops(model, input_dim, device='cuda'):
    """Estimates FLOPs for a single forward pass using thop or manual estimation."""
    try:
        from thop import profile, clever_format
        
        dummy_input = torch.randn(1, input_dim).to(device)
        
        model.eval()
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        return flops
    except ImportError:
        return _manual_flops_estimation(model, input_dim)
    except Exception:
        return _manual_flops_estimation(model, input_dim)


def _manual_flops_estimation(model, input_dim):
    """Manual FLOPs estimation based on linear layers."""
    total_flops = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            flops = 2 * module.in_features * module.out_features
            total_flops += flops
        elif isinstance(module, torch.nn.LayerNorm):
            flops = 5 * module.normalized_shape[0]
            total_flops += flops
    
    return total_flops if total_flops > 0 else None


def export_runtime_comparison_table(all_results, output_dir):
    """Exports consolidated runtime comparison table across all datasets."""
    import pandas as pd
    
    summary_rows = []
    
    for dataset_name, df in all_results.items():
        row = {
            'Dataset': dataset_name,
            'AUC-ROC': f"{df['TEST AUC-ROC'].mean():.4f}",
            'AP': f"{df['TEST AP'].mean():.4f}",
            'F1': f"{df['TEST F1-Score'].mean():.4f}",
            'Train Time (s)': f"{df['Total Time (s)'].mean():.2f}",
            'Inference (ms/sample)': f"{df['Inference Time per Sample (s)'].mean() * 1000:.4f}",
        }
        
        if 'GPU Memory Peak (MB)' in df.columns and not df['GPU Memory Peak (MB)'].isna().all():
            row['GPU Mem (MB)'] = f"{df['GPU Memory Peak (MB)'].mean():.1f}"
        else:
            row['GPU Mem (MB)'] = 'N/A'
        
        if 'FLOPs' in df.columns and not df['FLOPs'].isna().all():
            flops_mean = df['FLOPs'].mean()
            if flops_mean >= 1e9:
                row['FLOPs'] = f"{flops_mean/1e9:.2f}G"
            elif flops_mean >= 1e6:
                row['FLOPs'] = f"{flops_mean/1e6:.2f}M"
            else:
                row['FLOPs'] = f"{flops_mean:.0f}"
        else:
            row['FLOPs'] = 'N/A'
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    csv_path = output_dir / "runtime_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nRuntime comparison saved to: {csv_path}")
    
    latex_path = output_dir / "runtime_comparison.tex"
    latex_table = summary_df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(summary_df.columns) - 1))
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    return summary_df