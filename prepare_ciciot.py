import pandas as pd
import glob
import os
import numpy as np

INPUT_FOLDER = "/mnt/hdd/gpcc/datasets/CIC_IoT/csv_10sec" 
OUTPUT_FILE = "datasets/CIC_IoT_2023.csv"

TOTAL_BENIGN = 250000 
TOTAL_ATTACK = 100000 

def load_data_smart(file_pattern, total_samples, label_val, is_attack=False):
    files = [f for f in glob.glob(file_pattern, recursive=True) if os.path.isfile(f)]
    
    if not files:
        print(f"Nenhum arquivo .csv encontrado em: {file_pattern}")
        return pd.DataFrame()
    
    print(f"Found {len(files)} valid files in {os.path.basename(os.path.dirname(file_pattern))}.")
    
    dfs = []
    samples_collected = 0
    
    samples_per_file = int(total_samples / max(1, min(len(files), 20))) + 1
    
    for f in files[:20]:
        if samples_collected >= total_samples:
            break
            
        try:
            df_chunk = pd.read_csv(f, nrows=samples_per_file)
            
            if is_attack:
                if 'label_full' in df_chunk.columns:
                    df_chunk['attack_category'] = df_chunk['label_full']
                else:
                    df_chunk['attack_category'] = 'Unknown_Attack'

                df_chunk['label'] = 1
            else:
                df_chunk['attack_category'] = 'Benign'
                df_chunk['label'] = 0
            
            dfs.append(df_chunk)
            samples_collected += len(df_chunk)
            print(f"  -> Lido {os.path.basename(f)}: {len(df_chunk)} linhas")
            
        except Exception as e:
            print(f"  -> Erro ao ler {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    
    if len(full_df) > total_samples:
        full_df = full_df.sample(n=total_samples, random_state=42)
        
    return full_df

def main():
    print("--- Preparing CIC-IoT-2023 dataset (label_full fixed) ---")
    
    path_benign = os.path.join(INPUT_FOLDER, "**", "*benign*", "*.csv")
    if len(glob.glob(path_benign, recursive=True)) == 0:
         path_benign = os.path.join(INPUT_FOLDER, "benign", "*.csv")

    df_benign = load_data_smart(path_benign, TOTAL_BENIGN, label_val=0, is_attack=False)

    path_attack = os.path.join(INPUT_FOLDER, "**", "*attack*", "*.csv")
    if len(glob.glob(path_attack, recursive=True)) == 0:
         path_attack = os.path.join(INPUT_FOLDER, "attack", "*.csv")

    df_attack = load_data_smart(path_attack, TOTAL_ATTACK, label_val=1, is_attack=True)
    
    if df_benign.empty or df_attack.empty:
        print("\nCritical error: failed to load data. Check the paths.")
        return

    print("3. Consolidating...")
    df_final = pd.concat([df_benign, df_attack], ignore_index=True)

    # Remove metadata (IP, MAC, timestamps) and textual labels.
    cols_to_drop = [
        'flow_id', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'timestamp', 
        'timestamp_start', 'timestamp_end', 'device_name', 'device_mac',
        'label_full', 'label1', 'label2', 'label3', 'label4',
        'Unnamed: 0'
    ]

    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], errors='ignore')
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Saving to: {OUTPUT_FILE}")
    print(f"Final shape: {df_final.shape}")
    print(f"Distribution: {df_final['label'].value_counts().to_dict()}")
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("Done. Dataset cleaned and ready for training.")

if __name__ == "__main__":
    main()