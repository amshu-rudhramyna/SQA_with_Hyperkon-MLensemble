import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path
import time

def run_spectral_pruning(top_k=512):
    print(f"[Phase 1.2] Starting Spectral Pruning (Top {top_k} bands)...")
    data_dir = Path('data')
    hsi_path = data_dir / 'train_hsi_phase1.npz'
    
    if not hsi_path.exists():
        print("Error: train_hsi_phase1.npz not found.")
        return

    print("Loading data...")
    data = np.load(hsi_path)
    X = data['X'] # (N, 1290, 20, 20)
    y = data['y'] # (N, 4)
    idxs = data['idx']
    
    # Take mean over spatial dims for MI calculation
    X_1d = np.nanmean(X, axis=(2, 3))
    
    print(f"Calculating Mutual Information for {X_1d.shape[1]} bands...")
    # Clean NaNs for MI
    X_clean = np.nan_to_num(X_1d, 0.0)
    y_clean = np.nan_to_num(y, 0.0)
    
    # Aggregate importance across all 4 targets
    total_mi = np.zeros(X_1d.shape[1])
    for i in range(y.shape[1]):
        print(f"  Target {i}...")
        mi = mutual_info_regression(X_clean, y_clean[:, i], random_state=42)
        total_mi += mi
        
    # Get top K indices
    best_indices = np.argsort(total_mi)[-top_k:]
    best_indices = np.sort(best_indices) # Keep original spectral order
    
    print(f"Selected {len(best_indices)} bands. Saving pruned data...")
    X_pruned = X[:, best_indices, :, :]
    
    out_path = data_dir / 'train_hsi_pruned.npz'
    np.savez_compressed(out_path, X=X_pruned, y=y, idx=idxs, bands=best_indices)
    
    print(f"Pruned data saved to {out_path} (Shape: {X_pruned.shape})")
    print(f"Reduction: {100 * (1 - top_k/1290):.1f}% bandwidth saved.")

if __name__ == '__main__':
    run_spectral_pruning()
