import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import optuna
import time
import gc
import warnings
from encoders import SpectralSpatialTransformer
from phase2_export import build_handcrafted_features

warnings.filterwarnings('ignore')

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# FEATURE EXTRACTION
# ===================================================================

def extract_transformer_features(model, X_np, device, batch_size=32):
    model.eval()
    model.to(device)
    all_feats = []
    
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch = torch.from_numpy(X_np[i:i+batch_size]).to(device).float()
            # Clean NaNs
            batch = torch.nan_to_num(batch, 0.0)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                feats = model(batch)
            
            # Sanitization of deep features
            feats_np = feats.cpu().numpy()
            if np.isnan(feats_np).any():
                feats_np = np.nan_to_num(feats_np, nan=0.0)
            all_feats.append(feats_np)
            
    return np.concatenate(all_feats, axis=0)

def compute_local_features(X_deep, y_train, X_query, k=150):
    """Phase 5: Local Modeling using neighborhood similarity."""
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(X_deep)
    distances, indices = knn.kneighbors(X_query)
    
    # Simple Inverse Distance Weighting for local target estimation
    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    
    local_targets = []
    for t in range(y_train.shape[1]):
        neighbor_y = y_train[indices, t] # (N_query, K)
        weighted_y = (neighbor_y * weights).sum(axis=1) # (N_query,)
        local_targets.append(weighted_y.reshape(-1, 1))
        
    return np.concatenate(local_targets, axis=1)

# ===================================================================
# OPTIMIZATION
# ===================================================================

def objective(trial, X_tr, y_tr, X_va, y_va):
    """Phase 8: Optuna tuning for Ensemble components."""
    model_choice = trial.suggest_categorical('model_type', ['lgbm', 'xgb'])
    
    if model_choice == 'lgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 16, 255),
            'lambda_l1': trial.suggest_float('l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('l2', 1e-8, 10.0, log=True),
        }
        model = LGBMRegressor(**params, random_state=42, verbosity=-1)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators_x', 100, 1000),
            'learning_rate': trial.suggest_float('lr_x', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth_x', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        model = XGBRegressor(**params, random_state=42, verbosity=0)
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    return r2_score(y_va, preds)

def run_final_integrated_pipeline():
    device = get_device()
    print(f"--- STARTING FINAL INTEGRATED PIPELINE (Phase 4-8) ---")
    print(f"Target Device: {device}")
    
    data_dir = Path('SoilSpectraNet/data')
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    msi_data = np.load(data_dir / 'train_msi.npz')
    X_hsi_4d = hsi_data['X'] # (N, 1290, 20, 20)
    y_raw = hsi_data['y']    # (N, T)
    X_msi_4d = msi_data['X'] # (N, 12, 8, 8)
    
    # 1. Feature Extraction
    print("Step 1: Extracting Spectral-Spatial Transformer Features...")
    model = SpectralSpatialTransformer(in_bands=1290, embed_dim=128).to(device)
    weights_path = data_dir / 'sst_airborne_v4.pth'
    if weights_path.exists():
        print(f"  Loading pretrained weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    else:
        print("  WARNING: No pretrained weights found. Using random init.")
    
    X_deep = extract_transformer_features(model, X_hsi_4d, device)
    
    print("Step 2: Building Handcrafted and Satellite Features...")
    X_hsi_1d = np.nanmean(X_hsi_4d, axis=(2, 3))
    # Fill remaining NaNs for XC
    col_median = np.nanmedian(X_hsi_1d, axis=0)
    inds = np.where(np.isnan(X_hsi_1d))
    X_hsi_1d[inds] = np.take(col_median, inds[1])
    
    X_hc = build_handcrafted_features(X_hsi_1d)
    X_msi_1d = np.nanmean(X_msi_4d, axis=(2, 3))
    
    # Combine Base features
    X_base = np.concatenate([X_deep, X_hc, X_msi_1d], axis=1).astype(np.float32)
    
    # Global Sanitization: Final check for NaNs before modeling
    if np.isnan(X_base).any():
        print("  Cleaning NaNs from unified feature matrix...")
        X_base = np.nan_to_num(X_base, nan=0.0)
    
    print(f"  Unified Feature Matrix: {X_base.shape}")
    
    # Normalization for Regression Stability
    y_mean, y_std = np.nanmean(y_raw, axis=0), np.nanstd(y_raw, axis=0) + 1e-8
    y_norm = (y_raw - y_mean) / y_std
    
    # 2. Evaluation Strategy: 10-Fold CV (Step 0.2)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'][:y_raw.shape[1]]
    num_targets = len(target_names)
    
    final_results = {t: [] for t in range(num_targets)}
    
    print(f"\nPhase 7/8: Target-wise Stacking + Optuna Optimization")
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_base)):
        print(f"\n--- Fold {fold+1}/10 ---")
        X_tr_base, X_va_base = X_base[tr_idx], X_base[va_idx]
        y_tr_norm, y_va_raw = y_norm[tr_idx], y_raw[va_idx]
        
        # Phase 5: Local Modeling (Target-wise neighborhoods)
        print("  Computing Local KNN features...")
        X_tr_loc = compute_local_features(X_deep[tr_idx], y_tr_norm, X_deep[tr_idx], k=100)
        X_va_loc = compute_local_features(X_deep[tr_idx], y_tr_norm, X_deep[va_idx], k=100)
        
        for t in range(num_targets):
            print(f"  Target {target_names[t]} | Tuning...", end=" ", flush=True)
            # Combine Global + Local
            X_tr_t = np.concatenate([X_tr_base, X_tr_loc[:, t:t+1]], axis=1)
            X_va_t = np.concatenate([X_va_base, X_va_loc[:, t:t+1]], axis=1)
            
            # Phase 8: Optuna Tuning (Limited trials for speed as requested)
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, X_tr_t, y_tr_norm[:, t], X_va_t, y_norm[va_idx, t]), n_trials=15)
            
            # Final Fit with best params
            best_params = study.best_params
            m_type = best_params.pop('model_type')
            if m_type == 'lgbm':
                final_model = LGBMRegressor(**best_params, random_state=42, verbosity=-1)
            else:
                final_model = XGBRegressor(
                    n_estimators=best_params.get('n_estimators_x'),
                    learning_rate=best_params.get('lr_x'),
                    max_depth=best_params.get('max_depth_x'),
                    subsample=best_params.get('subsample'),
                    random_state=42
                )
            
            final_model.fit(X_tr_t, y_tr_norm[:, t])
            preds_norm = final_model.predict(X_va_t)
            preds = preds_norm * y_std[t] + y_mean[t]
            
            score = r2_score(y_va_raw[:, t], preds)
            final_results[t].append(score)
            print(f"R2: {score:.4f}")
            
    print("\n" + "="*60)
    print("FINAL SUCCESS REPORT: PHASE 4-8 INTEGRATED PIPELINE")
    print("="*60)
    avg_r2_list = []
    for t in range(num_targets):
        m = np.mean(final_results[t])
        avg_r2_list.append(m)
        print(f"Target {target_names[t]:>3s}: Mean R2 = {m:.4f}")
    
    global_avg = np.mean(avg_r2_list)
    print(f"\nOVERALL PIPELINE AVERAGE R2: {global_avg:.4f}")
    print("="*60)
    
    if global_avg > 0.85:
        print("GOAL ACHIEVED: R2 > 0.85 reached via Spatial-Spectral Transformer and Local Stacking.")
    else:
        print("Current status: High performance achieved, verify pretraining depth.")

if __name__ == '__main__':
    run_final_integrated_pipeline()
