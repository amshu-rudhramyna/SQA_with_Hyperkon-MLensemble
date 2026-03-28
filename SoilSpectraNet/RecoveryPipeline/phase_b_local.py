import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed
import warnings
import gc
import os

warnings.filterwarnings('ignore')

def compute_cosine_similarity_neighbors(X_train, X_query, k=150):
    """Step B1 & B2: Cosine Similarity + Top K Selection."""
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(X_train)
    distances, indices = knn.kneighbors(X_query)
    return indices, distances

def train_local_model_single(X_train, y_t_tr, m_tr, neighbor_indices, X_query_point, global_pred):
    """Callback for parallel local model training."""
    import warnings
    warnings.filterwarnings('ignore')
    
    valid_neighbor_mask = m_tr[neighbor_indices]
    valid_neighbors = neighbor_indices[valid_neighbor_mask]
    
    if len(valid_neighbors) > 15: # Minimum threshold for local signal
        X_train_local = X_train[valid_neighbors]
        y_train_local = y_t_tr[valid_neighbors]
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, 
                              num_leaves=15, random_state=42, verbosity=-1)
        model.fit(X_train_local, y_train_local)
        return model.predict(X_query_point.reshape(1, -1))[0]
    return global_pred

def run_phase_b_local():
    print(f"--- [PHASE B] LOCAL SIMILARITY MODELING ---")
    
    data_dir = Path('SoilSpectraNet/data')
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    X_hsi_4d = hsi_data['X'] # (N, 1290, 20, 20)
    y_raw = hsi_data['y']
    
    # Use raw spectral features for similarity (Step B1)
    X_spec = np.nanmean(X_hsi_4d, axis=(2, 3)).astype(np.float32)
    col_med = np.nanmedian(X_spec, axis=0)
    X_spec = np.where(np.isnan(X_spec), col_med, X_spec)
    
    # 10-Fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'][:y_raw.shape[1]]
    num_targets = len(target_names)
    
    final_blended_scores = {t: [] for t in range(num_targets)}
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_spec)):
        print(f"\nFold {fold+1}/10 | Processing Local Modeling...")
        X_tr, X_va = X_spec[tr_idx], X_spec[va_idx]
        y_tr, y_va = y_raw[tr_idx], y_raw[va_idx]
        
        # Step B1/B2: Find Top 150 Neighbors for each test point
        indices, _ = compute_cosine_similarity_neighbors(X_tr, X_va, k=150)
        
        for t in range(num_targets):
            y_t_tr, y_t_va = y_tr[:, t], y_va[:, t]
            m_tr, m_va = ~np.isnan(y_t_tr), ~np.isnan(y_t_va)
            
            if not m_tr.any() or not m_va.any(): continue
            
            # --- Global Model (from Phase A style) ---
            global_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42, verbosity=-1)
            global_model.fit(X_tr[m_tr], y_t_tr[m_tr])
            preds_global = global_model.predict(X_va[m_va])
            
            # --- Local Model (Step B3 - Parallelized) ---
            print(f"    Target {target_names[t]} | Parallel Local Fitting...", end=" ", flush=True)
            va_indices_with_target = np.where(m_va)[0]
            
            # Use Parallel processing (i9-12900K has 16+ cores)
            results = Parallel(n_jobs=-1)(delayed(train_local_model_single)(
                X_tr, y_t_tr, m_tr, indices[idx], X_va[idx], preds_global[i]
            ) for i, idx in enumerate(va_indices_with_target))
            
            preds_local = np.array(results)
            
            # Step B4: Weighted Blending (0.7 Global + 0.3 Local)
            preds_blended = 0.7 * preds_global + 0.3 * preds_local
            
            score = r2_score(y_t_va[m_va], preds_blended)
            final_blended_scores[t].append(score)
            print(f"R2: {score:.4f}")
            
    print("\n" + "="*50)
    print("PHASE B SUMMARY RESULTS (Local Blended)")
    print("="*50)
    for t in range(num_targets):
        m = np.mean(final_blended_scores[t])
        print(f"Target {target_names[t]:>3s}: Blended Mean R2 = {m:.4f}")
    
    total_avg = np.mean([np.mean(v) for v in final_blended_scores.values()])
    print(f"\nOVERALL RECOVERED R2 (PHASE B): {total_avg:.4f}")
    print("="*50)

if __name__ == '__main__':
    run_phase_b_local()
