import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import time

def get_base_models():
    return [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42)),
        ('svr', SVR(C=1.0, epsilon=0.1))
    ]

def get_stacking_ensemble():
    level0 = get_base_models()
    level1 = Ridge(alpha=1.0)
    return StackingRegressor(estimators=level0, final_estimator=level1, n_jobs=1)

def train_and_evaluate_baseline():
    """
    Phase 0.1 & 0.2: Freeze current best (CPU ensemble) and run 10-Fold CV.
    """
    print("Loading preprocessed data (SNV + MSC applied)...")
    data_dir = Path('data')
    
    # Let's use HSI Airborne as the primary baseline feature (flattened for the CPU models)
    hsi_data = np.load(data_dir / 'train_hsi.npz')
    
    X = hsi_data['X'] # (N, Bands, H, W)
    y = hsi_data['y'] # (N, Targets)
    
    # Flatten spatial dims to fallback to basic ML format Since we froze Approach 1
    # Actually, Approach 1 used pure Local KNN on late fusion. But here we just use the global ensemble
    # as the frozen baseline to establish 10-fold CV without the huge K-NN overhead.
    X_flat = np.mean(X, axis=(2, 3)) # Spatial mean pooling to (N, Bands)
    
    print(f"Dataset Shape: {X_flat.shape}")
    
    targets_count = y.shape[1]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    all_r2 = {t: [] for t in range(targets_count)}
    all_rmse = {t: [] for t in range(targets_count)}
    
    print("\n--- PHASE 0: 10-Fold CV Baseline Evaluation ---")
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_flat)):
        print(f"Training Fold {fold+1}/10...")
        X_train, X_val = X_flat[train_idx], X_flat[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for t in range(targets_count):
            model = get_stacking_ensemble()
            model.fit(X_train, y_train[:, t])
            preds = model.predict(X_val)
            
            r2 = r2_score(y_val[:, t], preds)
            rmse = np.sqrt(mean_squared_error(y_val[:, t], preds))
            
            all_r2[t].append(r2)
            all_rmse[t].append(rmse)
            
    print(f"\nCompleted in {time.time() - start_time:.2f} seconds.")
    
    # Log Phase 0.3 Metrics
    print("\n------------------------------------------------------------")
    print("PHASE 0 LOG: BASELINE METRICS (10-Fold CV)")
    print("------------------------------------------------------------")
    avg_r2_across_targets = []
    for t in range(targets_count):
        mean_r2 = np.mean(all_r2[t])
        mean_rmse = np.mean(all_rmse[t])
        avg_r2_across_targets.append(mean_r2)
        print(f"Target {t}: Average R2 = {mean_r2:.4f} | Average RMSE = {mean_rmse:.4f}")
        
    print(f"\nGLOBAL BASELINE AVERAGE R2: {np.mean(avg_r2_across_targets):.4f}")
    
if __name__ == '__main__':
    train_and_evaluate_baseline()
