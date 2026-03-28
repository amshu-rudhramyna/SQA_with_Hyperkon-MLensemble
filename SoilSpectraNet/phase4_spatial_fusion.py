import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import optuna
import time
import gc
from encoders import SpatialSpectral3DCNN, SpectralCNN, CrossAttentionFusion

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# FEATURE EXTRACTION ENGINE
# ===================================================================

class MultiModalFeatureExtractor:
    def __init__(self, hsi_bands, msi_bands, device):
        self.device = device
        self.hsi_encoder = SpatialSpectral3DCNN(in_bands=hsi_bands, out_dim=128).to(device)
        self.msi_encoder = SpectralCNN(in_bands=msi_bands, out_dim=64).to(device)
        self.fusion = CrossAttentionFusion(hsi_dim=128, msi_dim=64, embed_dim=128).to(device)
        
    def load_pretrained_hsi(self, path):
        print(f"Loading pretrained 3D encoder from {path}")
        self.hsi_encoder.load_state_dict(torch.load(path, weights_only=True))
        
    def extract(self, X_hsi, X_msi, batch_size=16):
        self.hsi_encoder.eval()
        self.msi_encoder.eval()
        self.fusion.eval()
        
        N = len(X_hsi)
        fused_features = []
        
        print(f"  Extracting features for {N} samples...")
        with torch.no_grad():
            for i in range(0, N, batch_size):
                b_hsi = torch.from_numpy(X_hsi[i:i+batch_size]).to(self.device).float()
                b_msi = torch.from_numpy(X_msi[i:i+batch_size]).to(self.device).float()
                
                # Input sanitation
                b_hsi = torch.nan_to_num(b_hsi, 0.0)
                b_msi = torch.nan_to_num(b_msi, 0.0)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    h_feat = self.hsi_encoder(b_hsi)
                    m_feat = self.msi_encoder(b_msi)
                    f_feat = self.fusion(h_feat, m_feat)
                
                # Collect and sanitize
                f_feat_np = f_feat.cpu().numpy()
                if np.isnan(f_feat_np).any():
                    f_feat_np = np.nan_to_num(f_feat_np, nan=0.0)
                    
                fused_features.append(f_feat_np)
                
        return np.concatenate(fused_features, axis=0)

# ===================================================================
# DATA LOADING
# ===================================================================

def load_all_data():
    data_dir = Path('data')
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    msi_data = np.load(data_dir / 'train_msi.npz')
    
    # Raw 4D HSI and 4D MSI
    X_hsi = hsi_data['X'] # (N, 1290, 20, 20)
    X_msi = msi_data['X'] # (N, 12, 8, 8)
    y = hsi_data['y']      # (N, 6)
    
    # 1D HSI for backward compatibility and Handcrafted
    X_hsi_1d = np.nanmean(X_hsi, axis=(2, 3))
    
    return X_hsi, X_msi, X_hsi_1d, y

# reuse phase 2/3 logic for handcrafted
from phase2_export import build_handcrafted_features

def compute_local_features(X_deep, y_train, X_query, k=200):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(X_deep)
    distances, indices = knn.kneighbors(X_query)
    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    
    local_feats = []
    for t in range(y_train.shape[1]):
        neighbor_y = y_train[indices, t]
        weighted_y = (neighbor_y * weights).sum(axis=1)
        local_feats.append(weighted_y.reshape(-1, 1))
    return np.concatenate(local_feats, axis=1)

# ===================================================================
# OPTIMIZATION & TRAINING
# ===================================================================

def objective(trial, X_tr, y_tr, X_va, y_va):
    model_type = trial.suggest_categorical('model', ['lgbm', 'xgb'])
    if model_type == 'lgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        }
        model = LGBMRegressor(**params, random_state=42, verbosity=-1)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators_x', 100, 600),
            'learning_rate': trial.suggest_float('lr_x', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth_x', 3, 10),
        }
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        
    model.fit(X_tr, y_tr)
    return r2_score(y_va, model.predict(X_va))

def run_phase_4():
    device = get_device()
    print(f"[Phase 4] Spatial Fusion + Optimized Ensemble | Device: {device}")
    
    X_hsi_4d, X_msi_4d, X_hsi_1d, y_raw = load_all_data()
    # Correcting target names to match 'y' shape (1876, 4)
    target_names = ['B', 'Cu', 'Zn', 'Fe']
    
    # 1. Extract Deep Features
    extractor = MultiModalFeatureExtractor(X_hsi_4d.shape[1], X_msi_4d.shape[1], device)
    if Path('data/encoder_3d_v4.pth').exists():
        extractor.load_pretrained_hsi('data/encoder_3d_v4.pth')
    
    print("Extracting Spatial-Spectral Fused features...")
    X_deep = extractor.extract(X_hsi_4d, X_msi_4d)
    
    # 2. Add Handcrafted (HSI 1D)
    X_hc = build_handcrafted_features(X_hsi_1d)
    
    # 3. Concatenate Base Features
    X_base = np.concatenate([X_deep, X_hc], axis=1).astype(np.float32)
    print(f"Base Feature Matrix: {X_base.shape} (Deep: {X_deep.shape[1]}, HC: {X_hc.shape[1]})")

    # Target-wise Normalization
    y_mean, y_std = np.mean(y_raw, axis=0), np.std(y_raw, axis=0) + 1e-8
    y_norm = (y_raw - y_mean) / y_std

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_scores = {t: [] for t in range(len(target_names))}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_base)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_tr_base, X_va_base = X_base[tr_idx], X_base[va_idx]
        y_tr_raw, y_va_raw = y_raw[tr_idx], y_raw[va_idx]
        y_tr_norm, y_va_norm = y_norm[tr_idx], y_norm[va_idx]
        
        # 4. Compute Local Similarity (Target-wise)
        print("  Computing Local (KNN) features...")
        X_tr_loc = compute_local_features(X_deep[tr_idx], y_tr_norm, X_deep[tr_idx])
        X_va_loc = compute_local_features(X_deep[tr_idx], y_tr_norm, X_deep[va_idx])
        
        for t in range(len(target_names)):
            print(f"  Optimizing Target {target_names[t]}...", end=" ")
            # Combine all features for this target
            X_tr = np.concatenate([X_tr_base, X_tr_loc[:, t:t+1]], axis=1)
            X_va = np.concatenate([X_va_base, X_va_loc[:, t:t+1]], axis=1)
            
            # Tune
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, X_tr, y_tr_norm[:, t], X_va, y_va_norm[:, t]), n_trials=20)
            
            # Final Fit with best params
            best_params = study.best_params
            model_type = best_params.pop('model')
            if model_type == 'lgbm':
                model = LGBMRegressor(**best_params, random_state=42, verbosity=-1)
            else:
                # remap keys if needed or just use consistent naming
                model = XGBRegressor(n_estimators=best_params.get('n_estimators_x'), 
                                     learning_rate=best_params.get('lr_x'),
                                     max_depth=best_params.get('max_depth_x'), random_state=42)
            
            model.fit(X_tr, y_tr_norm[:, t])
            preds = model.predict(X_va) * y_std[t] + y_mean[t]
            score = r2_score(y_va_raw[:, t], preds)
            final_scores[t].append(score)
            print(f"R2 = {score:.4f}")

    print("\n" + "="*50)
    print("PHASE 4 FINAL LOG: SPATIAL FUSION + TARGET-WISE OPTUNA")
    print("="*50)
    avg_r2 = []
    for t in range(len(target_names)):
        m = np.mean(final_scores[t])
        avg_r2.append(m)
        print(f"{target_names[t]:>3s}: Mean R2 = {m:.4f}")
    print(f"\nOVERALL PHASE 4 AVERAGE R2: {np.mean(avg_r2):.4f}")

if __name__ == '__main__':
    run_phase_4()
