"""
Data loading and preprocessing for CMAPSS dataset.
Includes:
- Data loading from CMAPSS files
- KMeans per-condition normalization
- Optional delta features
- Sequence windowing


Author: Aref Aasi
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from config import SEED, BASE_DIR


def load_cmapss_data(
    sub_dataset='FD002',
    window_size=80,
    rul_cap=125,
    data_dir=os.path.join(BASE_DIR, "CMAPSSData"),
    n_condition_clusters=8,
    add_deltas=True,
    kmeans_random_state=SEED
):
    """
    Load and preprocess CMAPSS dataset with per-condition normalization.
    
    Args:
        sub_dataset: Dataset name (e.g., 'FD002')
        window_size: Sequence window length
        rul_cap: Maximum RUL cap value
        data_dir: Directory containing CMAPSS data files
        n_condition_clusters: Number of KMeans clusters for operating conditions
        add_deltas: Whether to add first-difference features
        kmeans_random_state: Random state for KMeans
    
    Returns:
        X_train, y_train: Training sequences and labels
        X_test, y_test: Test sequences and labels
        n_features: Number of features
        window_size: Window size used
        test_unit_ids: Unit IDs for test set (for last-window evaluation)
    """
    try:
        train_file = os.path.join(data_dir, f"train_{sub_dataset}.txt")
        test_file  = os.path.join(data_dir, f"test_{sub_dataset}.txt")
        rul_file   = os.path.join(data_dir, f"RUL_{sub_dataset}.txt")
        
        if not all(os.path.exists(f) for f in [train_file, test_file, rul_file]):
            raise FileNotFoundError(f"Dataset files missing in {data_dir}")

        train_df = pd.read_csv(train_file, sep=r"\s+", header=None, engine="python").iloc[:, :26]
        test_df  = pd.read_csv(test_file,  sep=r"\s+", header=None, engine="python").iloc[:, :26]
        rul_test = pd.read_csv(rul_file,   header=None)

        columns = ['unit','cycle'] + [f'op{i}' for i in range(1,4)] + [f'sensor{i}' for i in range(1,22)]
        train_df.columns = columns
        test_df.columns  = columns

        # Base features
        sensor_cols = [f'sensor{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
        op_cols = [f'op{i}' for i in range(1,4)]
        norm_cols = op_cols + sensor_cols

        # Optional first-difference (delta) features per unit
        if add_deltas:
            for col in sensor_cols:
                train_df[f'd_{col}'] = train_df.groupby('unit')[col].diff().fillna(0.0)
                test_df[f'd_{col}']  = test_df.groupby('unit')[col].diff().fillna(0.0)
            delta_cols = [f'd_{c}' for c in sensor_cols]
            norm_cols = op_cols + sensor_cols + delta_cols

        # Train RUL (capped)
        max_cycles = train_df.groupby('unit')['cycle'].max().reset_index(name='max_cycle')
        train_df = train_df.merge(max_cycles, on='unit', how='left')
        train_df['rul'] = np.minimum(train_df['max_cycle'] - train_df['cycle'] + 1, rul_cap)

        # Test RUL
        test_max_cycles = test_df.groupby('unit')['cycle'].max().reset_index(name='max_cycle')
        test_df = test_df.merge(test_max_cycles, on='unit', how='left')
        test_units_sorted = np.sort(test_df['unit'].unique())
        
        if len(test_units_sorted) != len(rul_test):
            raise ValueError(f"RUL file length ({len(rul_test)}) != # test units ({len(test_units_sorted)})")
        
        unit_to_base_rul = dict(zip(test_units_sorted, rul_test[0].values))
        test_df['base_rul'] = test_df['unit'].map(unit_to_base_rul)
        test_df['rul'] = np.minimum(test_df['base_rul'] + (test_df['max_cycle'] - test_df['cycle']), rul_cap)

        # KMeans per-condition clustering on ops (train only)
        logging.info(f"KMeans on ops with k={n_condition_clusters}")
        kmeans = KMeans(n_clusters=n_condition_clusters, random_state=kmeans_random_state, n_init=10)
        kmeans.fit(train_df[op_cols].values)
        train_df['cond_cluster'] = kmeans.labels_
        test_df['cond_cluster']  = kmeans.predict(test_df[op_cols].values)

        # Per-cluster MinMax on TRAIN, apply to TRAIN/TEST
        scalers_by_cluster = {}
        for cl in sorted(train_df['cond_cluster'].unique()):
            idx = (train_df['cond_cluster'] == cl)
            scaler = MinMaxScaler().fit(train_df.loc[idx, norm_cols])
            scalers_by_cluster[cl] = scaler

        fallback_scaler = MinMaxScaler().fit(train_df[norm_cols])

        for cl, scaler in scalers_by_cluster.items():
            idx = (train_df['cond_cluster'] == cl)
            train_df.loc[idx, norm_cols] = scaler.transform(train_df.loc[idx, norm_cols])

        for cl in sorted(test_df['cond_cluster'].unique()):
            idx = (test_df['cond_cluster'] == cl)
            scaler = scalers_by_cluster.get(cl, fallback_scaler)
            test_df.loc[idx, norm_cols] = scaler.transform(test_df.loc[idx, norm_cols])

        # TRAIN: all windows; TEST: last window per unit (with ids)
        X_train, y_train = create_sequences(train_df, window_size, norm_cols, last_only=False)
        X_test,  y_test, test_unit_ids = create_sequences(test_df,  window_size, norm_cols, 
                                                          last_only=True, return_unit_ids=True)

        logging.info(f"Loaded: X_train={X_train.shape}, X_test={X_test.shape}, "
                    f"feats={len(norm_cols)}, window={window_size}, deltas={add_deltas}")
        
        return X_train, y_train, X_test, y_test, len(norm_cols), window_size, test_unit_ids

    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        print(f"Data loading failed: {e}")
        raise


def create_sequences(df, window, feature_cols, target_col='rul', last_only=False, return_unit_ids=False):
    """
    Create sliding window sequences from dataframe.
    
    Args:
        df: Input dataframe with 'unit' column
        window: Window size
        feature_cols: List of feature column names
        target_col: Target column name (default: 'rul')
        last_only: If True, only return last window per unit
        return_unit_ids: If True, return unit IDs along with sequences
    
    Returns:
        X: Array of sequences (N, window, n_features)
        y: Array of targets (N,)
        ids (optional): List of unit IDs
    """
    X, y, ids = [], [], []
    units = df['unit'].unique()
    
    for unit in units:
        u = df[df['unit'] == unit].sort_values('cycle')
        if len(u) < window:
            continue
            
        if last_only:
            start = len(u) - window
            end = len(u)
            X.append(u[feature_cols].iloc[start:end].values)
            y.append(u[target_col].iloc[end-1])
            if return_unit_ids:
                ids.append(unit)
        else:
            for start in range(len(u) - window + 1):
                end = start + window
                X.append(u[feature_cols].iloc[start:end].values)
                y.append(u[target_col].iloc[end-1])
    
    if return_unit_ids:
        return np.array(X), np.array(y), ids
    return np.array(X), np.array(y)
