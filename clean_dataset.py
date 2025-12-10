import pandas as pd

FILE_PATH = "datasets/CIC_IoT_2023.csv"

def clean():
    print("--- Cleaning CIC-IoT dataset ---")
    df = pd.read_csv(FILE_PATH)
    print(f"Shape original: {df.shape}")
    
    garbage_cols = [
        'log_data-types', 
        'network_ips_all', 'network_ips_dst', 'network_ips_src', 
        'network_macs_all', 'network_macs_dst', 'network_macs_src', 
        'network_ports_all', 'network_ports_dst', 'network_ports_src', 
        'network_protocols_all', 'network_protocols_dst', 'network_protocols_src'
    ]
    
    df = df.drop(columns=[c for c in garbage_cols if c in df.columns], errors='ignore')

    if 'attack_category' in df.columns:
        print("Saving attack_category to 'datasets/CIC_IoT_labels.csv' to avoid leakage during training...")
        df[['attack_category']].to_csv("datasets/CIC_IoT_labels.csv", index=False)
        df = df.drop(columns=['attack_category'])

    print(f"Shape final: {df.shape}")
    df.to_csv(FILE_PATH, index=False)
    print("Dataset cleaned. Ready to run the benchmark.")

if __name__ == "__main__":
    clean()