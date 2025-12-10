import pandas as pd
import os

FILE_PATH = "datasets/CIC_IoT_2023.csv"

def inspect():
    print(f"--- Inspecting: {FILE_PATH} ---")
    
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        print("   Make sure to run 'prepare_ciciot.py' first.")
        return

    try:
        df = pd.read_csv(FILE_PATH)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} cols")
    
    print("\nColumn list:")
    cols = df.columns.tolist()
    for i in range(0, len(cols), 5):
        print(f"   {cols[i:i+5]}")

    print("\nLabel distribution (0=Normal, 1=Attack):")
    if 'label' in df.columns:
        print(df['label'].value_counts())
    else:
        print("ERROR: 'label' column not found!")

    if 'attack_category' in df.columns:
        print("\nTop 20 attack types:")
        print(df['attack_category'].value_counts().head(20))
    else:
        print("\nWARNING: 'attack_category' column not found; cannot detail by attack type.")

    nulos = df.isnull().sum().sum()
    print(f"\nTotal NaN values: {nulos}")
    if nulos > 0:
        print("   Columns with NaNs:")
        print(df.columns[df.isnull().any()].tolist())

    print("\nFirst 5 rows:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head())

if __name__ == "__main__":
    inspect()