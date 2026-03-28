import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed
import os
import warnings
import gc

warnings.filterwarnings('ignore')

def compute_cosine_similarity_neighbors(X_train, X_query, k=150):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(X_train)
    distances, indices = knn.kneighbors(X_query)
    return indices, distances

def train_local_model_single(X_train, y_t_tr, m_tr, neighbor_indices, X_query_point, global_pred):
    import warnings
    warnings.filterwarnings('ignore')
    valid_neighbor_mask = m_tr[neighbor_indices]
    valid_neighbors = neighbor_indices[valid_neighbor_mask]
    if len(valid_neighbors) > 20: 
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, num_leaves=15, random_state=42, verbosity=-1)
        model.fit(X_train[valid_neighbors], y_t_tr[valid_neighbors])
        return model.predict(X_query_point.reshape(1, -1))[0]
    return global_pred

def compute_derivatives(X):
    d1 = np.diff(X, axis=1, prepend=X[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    return np.concatenate([X, d1, d2], axis=1)

def run_final_integrated_pipeline():
    print(f"--- [PHASE E] FINAL INTEGRATED CALIBRATION ---")
    data_dir = Path('../data')
    if not data_dir.exists(): data_dir = Path('SoilSpectraNet/data')
    
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    X_hsi_4d = hsi_data['X']
    y_raw = hsi_data['y']
    X_spec = np.nanmean(X_hsi_4d, axis=(2, 3)).astype(np.float32)
    X_spec = np.nan_to_num(X_spec, nan=np.nanmedian(X_spec))
    X_enriched = compute_derivatives(X_spec)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'][:y_raw.shape[1]]
    num_targets = len(target_names)
    
    final_logs = []
    total_scores = {t: [] for t in range(num_targets)}
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_spec)):
        print(f"\nProcessing Fold {fold+1}/10...")
        X_tr_spec, X_va_spec = X_spec[tr_idx], X_spec[va_idx]
        X_tr_enr, X_va_enr = X_enriched[tr_idx], X_enriched[va_idx]
        y_tr, y_va = y_raw[tr_idx], y_raw[va_idx]
        
        # 1. Similarity for local modeling
        indices, _ = compute_cosine_similarity_neighbors(X_tr_spec, X_va_spec, k=150)
        
        for t in range(num_targets):
            y_t_tr, y_t_va = y_tr[:, t], y_va[:, t]
            m_tr, m_va = ~np.isnan(y_t_tr), ~np.isnan(y_t_va)
            if not m_tr.any() or not m_va.any(): continue
            
            # --- Global Stack (Spectral + Derivatives) ---
            lgbm_g = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, verbosity=-1)
            lgbm_g.fit(X_tr_enr[m_tr], y_t_tr[m_tr])
            p_global = lgbm_g.predict(X_va_enr[m_va])
            
            # --- Local Modeling ---
            va_inds = np.where(m_va)[0]
            p_local = Parallel(n_jobs=-1)(delayed(train_local_model_single)(
                X_tr_spec, y_t_tr, m_tr, indices[idx], X_va_spec[idx], p_global[i]
            ) for i, idx in enumerate(va_inds))
            p_local = np.array(p_local)
            
            # --- Weighted Blending (0.7 G + 0.3 L) ---
            preds = 0.7 * p_global + 0.3 * p_local
            score = r2_score(y_t_va[m_va], preds)
            total_scores[t].append(score)
            log_line = f"Fold {fold+1} | {target_names[t]}: {score:.4f}"
            print(f"  {log_line}")
            final_logs.append(log_line)
            
    # Success Summary
    print("\n" + "="*50)
    print("FINAL INTEGRATED SUCCESS REPORT")
    print("="*50)
    final_logs.append("\nFINAL INTEGRATED SUCCESS REPORT")
    avg_r2 = []
    for t in range(num_targets):
        m = np.mean(total_scores[t])
        avg_r2.append(m)
        sum_line = f"Target {target_names[t]:>3s}: CALIBRATED R2 = {m:.4f}"
        print(sum_line)
        final_logs.append(sum_line)
    
    overall = np.mean(avg_r2)
    final_logs.append(f"\nOVERALL MODEL R2: {overall:.4f}")
    
    # Save log.txt
    log_file = Path('log.txt')
    with open(log_file, 'w') as f:
        f.write("\n".join(final_logs))
    print(f"Detailed logs saved to {log_file.absolute()}")

if __name__ == '__main__':
    run_final_integrated_pipeline()
