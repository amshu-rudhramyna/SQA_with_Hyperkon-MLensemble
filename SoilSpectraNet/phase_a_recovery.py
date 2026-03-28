import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
import gc
from encoders import ShallowSpectralCNN1D

warnings.filterwarnings('ignore')

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_shallow_cnn(X_tr, y_tr, X_va, y_va, device, epochs=30, batch_size=32):
    """Trains the shallow CNN on a per-fold basis to extract features."""
    model = ShallowSpectralCNN1D(in_bands=X_tr.shape[1], out_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Simple Tensor Datasets
    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    val_ds = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for i_x, i_y in train_loader:
            i_x, i_y = i_x.to(device), i_y.to(device)
            optimizer.zero_grad()
            preds = model(i_x)
            loss = criterion(preds, i_y)
            loss.backward()
            optimizer.step()
            
    # Extraction mode: return the model
    model.eval()
    return model

def extract_features(model, X, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        feats = model(X_t)
    return feats.cpu().numpy()

def run_phase_a_recovery():
    device = get_device()
    print(f"--- STARTING PHASE A: SIGNAL RECOVERY ---")
    data_dir = Path('SoilSpectraNet/data')
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    X_hsi_4d = hsi_data['X'] # (N, 1290, 20, 20)
    y_raw = hsi_data['y']    # (N, T)
    
    # 1. Step A1: Revert to 1D Spectral Data (Bands + Preprocessing outputs)
    # X_hsi_1d is the mean spectrum per sample
    X_spec = np.nanmean(X_hsi_4d, axis=(2, 3))
    # Replace NaNs with median
    col_median = np.nanmedian(X_spec, axis=0)
    inds = np.where(np.isnan(X_spec))
    X_spec[inds] = np.take(col_median, inds[1])
    
    print(f"  Base Spectral Shape: {X_spec.shape}")
    
    # 2. Setup Evaluation (10-Fold CV)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'][:y_raw.shape[1]]
    num_targets = len(target_names)
    
    cv_results = {t: [] for t in range(num_targets)}
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_spec)):
        print(f"\n--- Fold {fold+1}/10 ---")
        X_tr, X_va = X_spec[tr_idx], X_spec[va_idx]
        y_tr, y_va = y_raw[tr_idx], y_raw[va_idx]
        
        # Step A2/A3: Train shallow CNN per target then concatenate
        # For efficiency in Phase A, we train one CNN on the multi-output target first
        # to get a general feature extractor.
        print("  Training Shallow CNN for feature extraction...")
        # Clean targets for CNN
        y_tr_clean = np.nan_to_num(y_tr, nan=0.0)
        y_va_clean = np.nan_to_num(y_va, nan=0.0)
        
        cnn_model = train_shallow_cnn(X_tr, y_tr_clean, X_va, y_va_clean, device)
        X_tr_cnn = extract_features(cnn_model, X_tr, device)
        X_va_cnn = extract_features(cnn_model, X_va, device)
        
        # Concatenate Features (Step A3)
        X_tr_final = np.concatenate([X_tr, X_tr_cnn], axis=1)
        X_va_final = np.concatenate([X_va, X_va_cnn], axis=1)
        
        # Step A4: Train Ensemble (Target-wise)
        for t in range(num_targets):
            # Target-specific clean-up
            y_tr_t = y_tr[:, t]
            y_va_t = y_va[:, t]
            mask_tr = ~np.isnan(y_tr_t)
            mask_va = ~np.isnan(y_va_t)
            
            if not mask_tr.any() or not mask_va.any():
                continue
                
            # LightGBM (Primary)
            lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42, verbosity=-1)
            lgbm.fit(X_tr_final[mask_tr], y_tr_t[mask_tr])
            preds_lgbm = lgbm.predict(X_va_final[mask_va])
            
            # XGBoost (Secondary)
            xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
            xgb.fit(X_tr_final[mask_tr], y_tr_t[mask_tr])
            preds_xgb = xgb.predict(X_va_final[mask_va])
            
            # Simple average ensemble
            preds = (preds_lgbm + preds_xgb) / 2.0
            
            score = r2_score(y_va_t[mask_va], preds)
            cv_results[t].append(score)
            print(f"    Target {target_names[t]}: R2 = {score:.4f}")
            
    print("\n" + "="*50)
    print("PHASE A SUMMARY RESULTS")
    print("="*50)
    for t in range(num_targets):
        print(f"Target {target_names[t]:>3s}: Mean R2 = {np.mean(cv_results[t]):.4f}")
    print(f"Overall Mean R2: {np.mean([np.mean(v) for v in cv_results.values()]):.4f}")
    print("="*50)

if __name__ == '__main__':
    run_phase_a_recovery()
