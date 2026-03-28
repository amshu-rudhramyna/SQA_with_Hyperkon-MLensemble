import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sys
import os
import warnings
import gc

# Add parent directory to sys.path to allow imports from SoilSpectraNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoders import SpectralSpatialTransformer, ShallowSpectralCNN1D

warnings.filterwarnings('ignore')

def get_device():
    return torch.device('cuda' if torch.device('cuda' if torch.cuda.is_available() else 'cpu') else 'cpu')

def extract_transformer_features(model, X_hsi_4d, device, batch_size=32):
    """Step C1: Extracting low-dim transformer features."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(X_hsi_4d), batch_size):
            batch = torch.from_numpy(X_hsi_4d[i:i+batch_size]).float().to(device)
            # Standard normalization for transformer
            batch = torch.nan_to_num(batch, 0.0)
            with torch.amp.autocast('cuda'):
                feats = model(batch)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)

def compute_derivatives(X):
    d1 = np.diff(X, axis=1, prepend=X[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    return np.concatenate([X, d1, d2], axis=1)

def run_phase_cd_optimization():
    device = get_device()
    print(f"--- [PHASE C/D] TARGET-WISE OPTIMIZATION & FUSION ---")
    
    data_dir = Path('../data')
    if not data_dir.exists(): data_dir = Path('SoilSpectraNet/data')
    
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    X_hsi_4d = hsi_data['X']
    y_raw = hsi_data['y']
    
    # 1. Base Spectral Features + Derivatives
    X_spec = np.nanmean(X_hsi_4d, axis=(2, 3)).astype(np.float32)
    col_med = np.nanmedian(X_spec, axis=0)
    X_spec = np.where(np.isnan(X_spec), col_med, X_spec)
    X_enriched = compute_derivatives(X_spec)
    
    # 2. Extract Low-Dim Transformer Features (Step C1-C3)
    print("  Extracting Low-Dim Transformer auxiliary features...")
    sst = SpectralSpatialTransformer(in_bands=1290, embed_dim=128, out_dim=32).to(device)
    # Load pretrained weights if available
    weights_path = data_dir / 'sst_airborne_v4.pth'
    if weights_path.exists():
        try:
            # We need to handle the potentially different out_dim in the saved state
            state_dict = torch.load(weights_path, weights_only=True)
            # Filter out mlp_head weights due to out_dim change
            state_dict = {k: v for k, v in state_dict.items() if 'mlp_head' not in k}
            sst.load_state_dict(state_dict, strict=False)
            print("    Pretrained weights loaded (Backbone only).")
        except:
            print("    Weight mismatch detected. Using random initializer.")
    
    X_sst = extract_transformer_features(sst, X_hsi_4d, device)
    
    # 3. Target-Wise Evaluation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'][:y_raw.shape[1]]
    num_targets = len(target_names)
    
    final_cd_scores = {t: [] for t in range(num_targets)}
    
    # Pre-concatenate base features
    X_base = np.concatenate([X_enriched, X_sst], axis=1) # (N, 3870 + 32)
    # Step A1/C4: Ensure total NaN suppression before SelectKBest
    X_base = np.nan_to_num(X_base, nan=0.0)
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_base)):
        print(f"\nFold {fold+1}/10 Optimization Progress:")
        X_tr_fold, X_va_fold = X_base[tr_idx], X_base[va_idx]
        y_tr, y_va = y_raw[tr_idx], y_raw[va_idx]
        
        for t in range(num_targets):
            y_t_tr, y_t_va = y_tr[:, t], y_va[:, t]
            m_tr, m_va = ~np.isnan(y_t_tr), ~np.isnan(y_t_va)
            if not m_tr.any() or not m_va.any(): continue
            
            # Step D3: Target-Wise Band Selection
            # Select 800 most relevant features for THIS target
            selector = SelectKBest(score_func=f_regression, k=800)
            selector.fit(X_tr_fold[m_tr], y_t_tr[m_tr])
            X_tr_t = selector.transform(X_tr_fold[m_tr])
            X_va_t = selector.transform(X_va_fold[m_va])
            
            # Step D2: Target-Wise Models with high regularization
            # LightGBM (Primary)
            lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=7, 
                                 num_leaves=31, lambda_l1=0.1, lambda_l2=0.5,
                                 random_state=42, verbosity=-1)
            lgbm.fit(X_tr_t, y_t_tr[m_tr])
            p_lgbm = lgbm.predict(X_va_t)
            
            # XGBoost (Secondary)
            xgb = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, 
                               reg_alpha=0.1, reg_lambda=0.5,
                               random_state=42, n_jobs=-1)
            xgb.fit(X_tr_t, y_t_tr[m_tr])
            p_xgb = xgb.predict(X_va_t)
            
            # Blended Prediction
            preds = (p_lgbm * 0.6 + p_xgb * 0.4)
            
            score = r2_score(y_t_va[m_va], preds)
            final_cd_scores[t].append(score)
            print(f"    {target_names[t]}: {score:.4f}", end=" ", flush=True)
            
    print("\n" + "="*50)
    print("PHASE C/D FINAL SUCCESS REPORT")
    print("="*50)
    avg_list = []
    for t in range(num_targets):
        m = np.mean(final_cd_scores[t])
        avg_list.append(m)
        print(f"Target {target_names[t]:>3s}: Optimized Mean R2 = {m:.4f}")
    
    total_avg = np.mean(avg_list)
    print(f"\nOVERALL MODEL R2 (PHASE C/D): {total_avg:.4f}")
    if total_avg > 0.65:
        print("GOAL PROXIMITY: HIGH. Ready for final ensemble calibration.")
    print("="*50)

if __name__ == '__main__':
    run_phase_cd_optimization()
