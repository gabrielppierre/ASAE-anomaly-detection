import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import os

def preprocess_data(file_path, scaler_type='minmax', random_state=42, 
                   temporal_aware=False, dataset_fraction=1.0):
    """Loads, cleans, splits, and preprocesses data for anomaly detection."""
    print("  Preprocessing data...")
    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)
    
    if dataset_fraction < 1.0:
        if temporal_aware:
            print(f"    Using first {dataset_fraction*100:.0f}% of dataset (temporal)")
            num_rows = int(len(df) * dataset_fraction)
            df = df.iloc[:num_rows]
        else:
            print(f"    Using random {dataset_fraction*100:.0f}% of dataset")
            df = df.sample(frac=dataset_fraction, random_state=random_state)

    df.reset_index(drop=True, inplace=True)

    label_col_name = None
    possible_names = ['Label', 'label', 'attack_cat', 'Is_Attack', 'Class']
    for name in possible_names:
        if name in df.columns:
            label_col_name = name
            break
    
    if label_col_name is None:
        raise ValueError(f"Could not find label column in {os.path.basename(file_path)}")

    df.rename(columns={label_col_name: 'Label'}, inplace=True)

    if df['Label'].dtype == 'object':
        normal_labels = ['Normal', 'normal', 'Benign', 'benign', 0]
        df['Label'] = df['Label'].apply(lambda x: 0 if x in normal_labels else 1)
    
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
    df.dropna(subset=['Label'], inplace=True)

    X = df.drop('Label', axis=1)
    y = df['Label']

    cols_to_drop = ['srcip', 'dstip', 'attack_cat', 'timestamp']
    existing_cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    if existing_cols_to_drop:
        X = X.drop(columns=existing_cols_to_drop)

    port_cols = ['sport', 'dsport']
    for col in port_cols:
        if col in X.columns and X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)
    if temporal_aware:
        print("    Split: Temporal-Aware (train=benign only)")
        
        normal_indices = y[y == 0].index
        anomaly_indices = y[y == 1].index
        
        X_normal = X.loc[normal_indices]
        y_normal = y.loc[normal_indices]
        
        X_anomaly = X.loc[anomaly_indices]
        y_anomaly = y.loc[anomaly_indices]

        train_end_idx = int(len(X_normal) * 0.8)
        
        X_train = X_normal.iloc[:train_end_idx]
        y_train = y_normal.iloc[:train_end_idx]
        
        X_normal_holdout = X_normal.iloc[train_end_idx:]
        y_normal_holdout = y_normal.iloc[train_end_idx:]

        X_val_normal, X_test_normal, y_val_normal, y_test_normal = train_test_split(
            X_normal_holdout, y_normal_holdout, test_size=0.5, random_state=random_state
        )
        
        if len(y_anomaly) > 1:
            X_val_anomaly, X_test_anomaly, y_val_anomaly, y_test_anomaly = train_test_split(
                X_anomaly, y_anomaly, test_size=0.5, random_state=random_state, stratify=y_anomaly
            )
        else:
             X_val_anomaly, X_test_anomaly, y_val_anomaly, y_test_anomaly = train_test_split(
                X_anomaly, y_anomaly, test_size=0.5, random_state=random_state
            )

        X_val = pd.concat([X_val_normal, X_val_anomaly])
        y_val = pd.concat([y_val_normal, y_val_anomaly])
        
        X_test = pd.concat([X_test_normal, X_test_anomaly])
        y_test = pd.concat([y_test_normal, y_test_anomaly])
        
        X_val = X_val.sample(frac=1, random_state=random_state)
        y_val = y_val.loc[X_val.index]
        
        X_test = X_test.sample(frac=1, random_state=random_state)
        y_test = y_test.loc[X_test.index]

    else:
        print("    Split: Random Stratified (80/10/10)")
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=y, random_state=random_state
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=True, stratify=y_temp, random_state=random_state
        )

    print(f"      Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    X_train_np = X_train.to_numpy()
    X_val_np = X_val.to_numpy()
    X_test_np = X_test.to_numpy()

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_np)
    X_val_imputed = imputer.transform(X_val_np)
    X_test_imputed = imputer.transform(X_test_np)

    if scaler_type == 'minmax':
        scaler = MinMaxScaler(clip=True)
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return (
        X_train_scaled,
        y_train.values,
        X_val_scaled,
        y_val.values,
        X_test_scaled,
        y_test.values,
        scaler
    )