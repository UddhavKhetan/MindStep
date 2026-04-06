import numpy as np
import pandas as pd

def extract_features_from_window(window_df: pd.DataFrame) -> np.ndarray:
    """
    Extracts statistical features from a time window of IMU data.
    Requires window_df to contain ['ax', 'ay', 'az', 'gx', 'gy', 'gz'].
    Returns a 1D numpy array of length 30 (6 axes * 5 features).
    """
    cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    features = []
    
    for col in cols:
        data = window_df[col].values
        features.extend([
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.mean(data ** 2)  # Energy
        ])
        
    return np.array(features)

def build_windowed_dataset(df: pd.DataFrame, 
                           window_size: int = 10, 
                           step_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset of feature vectors and labels using a sliding window.
    
    df: Dataframe with columns ['time', 'ped_id', 'ax', ..., 'intent']
    window_size: Number of consecutive samples per window.
    step_size: Number of samples to slide the window.
    
    Returns (X, y):
      X: 2D numpy array of shape (n_windows, n_features)
      y: 1D numpy array of binary intent labels
    """
    X, y = [], []
    
    for ped_id, group in df.groupby("ped_id"):
        group = group.sort_values("time").reset_index(drop=True)
        n_samples = len(group)
        
        for i in range(0, n_samples - window_size + 1, step_size):
            window = group.iloc[i : i + window_size]
            features = extract_features_from_window(window)
            label = window["intent"].iloc[-1]
            
            X.append(features)
            y.append(label)
            
    return np.array(X), np.array(y)