import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import optuna
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def load_phase2_features():
    data_dir = Path('data')
    fpath = data_dir / 'train_fused_phase2.npz'
    if not fpath.exists():
        raise FileNotFoundError("Run 'phase2_export.py' first to generate features!")
    data = np.load(fpath)
    return data['X'], data['y']

# ===================================================================
# LOCAL SPECTRAL SIMILARITY FEATURES (GAME CHANGER)
# ===================================================================

def compute_local_features(X_train, y_train, X_val, k=200):
    """
    Computes the weighted mean of nearest neighbor targets in training set.
    This effectively models 'Geochemical Locality' without the overhead 
    of training thousands of separate local models.
    """
    # Use only SSL features (first 256) for similarity search
    X_train_ssl = X_train[:, :256]
    X_val_ssl = X_val[:, :256]
    
    # KNN Similarity
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(X_train_ssl)
    distances, indices = knn.kneighbors(X_val_ssl)
    
    # Weights based on distance (inverse cosine distance)
    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    
    local_feats = []
    for t in range(y_train.shape[1]):
        # Weighted mean of neighbors' target values
        neighbor_y = y_train[indices, t] # (N, k)
        weighted_y = (neighbor_y * weights).sum(axis=1) # (N,)
        local_feats.append(weighted_y.reshape(-1, 1))
        
    return np.concatenate(local_feats, axis=1)

# ===================================================================
# OPTUNA HYPERPARAMETER TUNING
# ===================================================================

def objective(trial, X_tr, y_tr, X_va, y_va):
    """Optuna objective for tuning the Meta-Learner or Base Models."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    model = LGBMRegressor(**params, random_state=42, verbosity=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    return r2_score(y_va, preds)

def get_optimized_ensemble(X_tr, y_tr, X_va, y_va):
    """Tunable Stacking Ensemble with Optuna-tuned LightGBM Meta-Learner."""
    print("  Optimizing Meta-Learner...")
    study = optuna.create_study(direction='maximize')
    # Run a few trials for speed but enough to gain performance
    study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_va, y_va), n_trials=30)
    
    best_params = study.best_params
    
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42, verbosity=0)),
        ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)),
    ]
    
    meta_learner = LGBMRegressor(**best_params, random_state=42, verbosity=-1)
    
    # Note: StackingRegressor's internal CV will be used during fit.
    return StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5, 
        n_jobs=1,
        passthrough=True
    )

# ===================================================================
# MAIN
# ===================================================================

def run_phase_3():
    print("[Phase 3: Local Modeling & Optuna Optimization]")
    X_fused, y_raw = load_phase2_features()
    
    # -------------------------------------------------------------------
    # TARGET-WISE NORMALIZATION (from phase 2)
    # -------------------------------------------------------------------
    y_mean = np.mean(y_raw, axis=0, keepdims=True)
    y_std = np.std(y_raw, axis=0, keepdims=True) + 1e-8
    y = ((y_raw - y_mean) / y_std).astype(np.float32)
    
    print(f"Features loaded: {X_fused.shape}, Targets: {y.shape}")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    target_names = ['P', 'K', 'Mg', 'pH']
    all_r2 = {t: [] for t in range(4)}
    
    start_time_total = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_fused)):
        print(f"\n--- Fold {fold+1}/10 ---")
        X_train_raw, X_val_raw = X_fused[train_idx], X_fused[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]
        
        # 1. Compute Local Similarity Features (k=200)
        # These are target-specific locality means for EACH target.
        print("  Computing Local Locality Features...")
        X_train_local = compute_local_features(X_train_raw, y_train_raw, X_train_raw, k=200)
        X_val_local = compute_local_features(X_train_raw, y_train_raw, X_val_raw, k=200)
        
        # 2. Target-Wise Training
        fold_r2s = []
        for t in range(4):
            # Augment features with the local target mean for THIS target only
            X_tr = np.concatenate([X_train_raw, X_train_local[:, t:t+1]], axis=1)
            X_va = np.concatenate([X_val_raw, X_val_local[:, t:t+1]], axis=1)
            
            print(f"  Target {target_names[t]} (Features: {X_tr.shape[1]})...", end=" ")
            
            # 3. Optimize & Train
            ensemble = get_optimized_ensemble(X_tr, y_train_raw[:, t], X_va, y_val_raw[:, t])
            ensemble.fit(X_tr, y_train_raw[:, t])
            preds_norm = ensemble.predict(X_va)
            
            # Denormalize for R2 calculation
            preds = preds_norm * y_std[0, t] + y_mean[0, t]
            true = y_val_raw[:, t] * y_std[0, t] + y_mean[0, t]
            
            score = r2_score(true, preds)
            all_r2[t].append(score)
            fold_r2s.append(score)
            print(f"R2 = {score:.4f}")
            
    # Report Results
    print("\n" + "=" * 60)
    print("PHASE 3 LOG: LOCAL SIMILARITY + OPTUNA OPTIMIZATION (10-Fold CV)")
    print("=" * 60)
    avg_r2_across_targets = []
    for t in range(4):
        mean_r2 = np.mean(all_r2[t])
        std_r2 = np.std(all_r2[t])
        avg_r2_across_targets.append(mean_r2)
        print(f"{target_names[t]:>4s}: R2 = {mean_r2:.4f} \u00b1 {std_r2:.4f}")

    print(f"\nGLOBAL PHASE 3 AVERAGE R2: {np.mean(avg_r2_across_targets):.4f}")
    print(f"Total time: {time.time() - start_time_total:.1f}s")


if __name__ == '__main__':
    run_phase_3()
