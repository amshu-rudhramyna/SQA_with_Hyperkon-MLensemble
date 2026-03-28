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
import sys
import os
import warnings
import gc

# Add parent directory to sys.path to allow imports from SoilSpectraNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoders import ShallowSpectralCNN1D

warnings.filterwarnings('ignore')

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_shallow_cnn(X_tr, y_tr, device, epochs=40, batch_size=64):
    """Phase A - Step A2: Shallow 1D CNN training."""
    num_targets = y_tr.shape[1]
    # We want 64 features, but we need to train on 'num_targets'
    model = ShallowSpectralCNN1D(in_bands=X_tr.shape[1], out_dim=64).to(device)
    # Add a temporary projection for the training objective
    train_head = nn.Linear(64, num_targets).to(device)
    
    optimizer = optim.AdamW(list(model.parameters()) + list(train_head.parameters()), lr=1e-3, weight_decay=1e-2)
    criterion = nn.MSELoss()
    
    y_tr_clean = np.nan_to_num(y_tr, nan=0.0)
    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr_clean).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for i_x, i_y in train_loader:
            i_x, i_y = i_x.to(device), i_y.to(device)
            optimizer.zero_grad()
            feats = model(i_x)
            preds = train_head(feats)
            loss = criterion(preds, i_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

def extract_features(model, X, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        feats = model(X_t)
    return feats.cpu().numpy()

def apply_feature_selection(X_ptr, y_ptr, n_top=1000):
    """Partial Step D3: Fast variance/correlation-based selection."""
    from sklearn.feature_selection import SelectKBest, f_regression
    y_clean = np.nan_to_num(y_ptr, nan=0.0)
    # Sum of correlations across all targets to find globally useful bands
    selector = SelectKBest(score_func=f_regression, k=n_top)
    # We use a single target or mean target for selection to save time
    selector.fit(X_ptr, np.mean(y_clean, axis=1))
    return selector

def compute_derivatives(X):
    """Step A1: Adding 1st and 2nd derivatives for stronger spectral signal."""
    # X: (N, Bands)
    d1 = np.diff(X, axis=1, prepend=X[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    return np.concatenate([X, d1, d2], axis=1)

def run_phase_a_recovery():
    device = get_device()
    print(f"--- [PHASE A] RECOVERING SIGNAL (RecoveryPipeline) ---")
    print(f"Device: {device}")
    
    data_dir = Path('../data') # Relative to RecoveryPipeline
    if not data_dir.exists(): data_dir = Path('SoilSpectraNet/data')
    
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    X_hsi_4d = hsi_data['X'] # (N, 1290, 20, 20)
    y_raw = hsi_data['y']    # (N, T)
    
    # 1. Step A1: Revert to 1D Spectral Data
    X_spec = np.nanmean(X_hsi_4d, axis=(2, 3)).astype(np.float32)
    # Median Imputation for NaNs
    col_med = np.nanmedian(X_spec, axis=0)
    X_spec = np.where(np.isnan(X_spec), col_med, X_spec)
    
    # Add Derivatives (X + D1 + D2) -> Triple the features
    X_enriched = compute_derivatives(X_spec)
    print(f"  Enriched Spectral Matrix: {X_enriched.shape}")
    
    # 2. 10-Fold CV Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'][:y_raw.shape[1]]
    num_targets = len(target_names)
    
    final_scores = {t: [] for t in range(num_targets)}
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_enriched)):
        print(f"\nFold {fold+1}/10 Progress:", end=" ", flush=True)
        X_tr, X_va = X_enriched[tr_idx], X_enriched[va_idx]
        y_tr, y_va = y_raw[tr_idx], y_raw[va_idx]
        
        # Step A2/A3: Shallow CNN features
        cnn_model = train_shallow_cnn(X_tr, y_tr, device, epochs=20) # Faster
        X_tr_cnn = extract_features(cnn_model, X_tr, device)
        X_va_cnn = extract_features(cnn_model, X_va, device)
        
        X_tr_final_all = np.concatenate([X_tr, X_tr_cnn], axis=1)
        X_va_final_all = np.concatenate([X_va, X_va_cnn], axis=1)
        
        # Feature Selection to reduce 3870+64 down to 1000 for speed (Step D3 partial)
        selector = apply_feature_selection(X_tr_final_all, y_tr, n_top=1000)
        X_tr_final = selector.transform(X_tr_final_all)
        X_va_final = selector.transform(X_va_final_all)
        
        # Stacking Ensemble per Target (Step A4)
        for t in range(num_targets):
            y_t_tr, y_t_va = y_tr[:, t], y_va[:, t]
            m_tr, m_va = ~np.isnan(y_t_tr), ~np.isnan(y_t_va)
            
            if not m_tr.any() or not m_va.any(): continue
            
            # Primary: LightGBM
            lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, 
                                 num_leaves=31, random_state=42, verbosity=-1)
            lgbm.fit(X_tr_final[m_tr], y_t_tr[m_tr])
            p_lgbm = lgbm.predict(X_va_final[m_va])
            
            # Secondary: XGBoost
            xgb = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, 
                               random_state=42, n_jobs=-1)
            xgb.fit(X_tr_final[m_tr], y_t_tr[m_tr])
            p_xgb = xgb.predict(X_va_final[m_va])
            
            preds = (p_lgbm + p_xgb) / 2.0
            score = r2_score(y_t_va[m_va], preds)
            final_scores[t].append(score)
            print(f"{target_names[t]}:{score:.2f}", end=" ", flush=True)
            
    print("\n\n" + "-"*40)
    print("PHASE A RECOVERY SUCCESS REPORT")
    print("-"*40)
    avg_r2 = []
    for t in range(num_targets):
        m = np.mean(final_scores[t])
        avg_r2.append(m)
        print(f"Target {target_names[t]:>3s}: Mean R2 = {m:.4f}")
    
    total_avg = np.mean(avg_r2)
    print(f"\nRECOVERED OVERALL R2: {total_avg:.4f}")
    if total_avg > 0.50:
        print("STATUS: SIGNAL RECOVERED. Ready for Phase B Local Modeling.")
    print("-"*40)

if __name__ == '__main__':
    run_phase_a_recovery()
