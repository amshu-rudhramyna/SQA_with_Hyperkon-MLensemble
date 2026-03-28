import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
import time
import warnings
warnings.filterwarnings('ignore')

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

def train_and_evaluate_phase1():
    print("Loading Phase 1 Representation Built Data...")
    data_dir = Path('SoilSpectraNet/data')
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    
    X = hsi_data['X'] # (N, 1290, 20, 20)
    y = hsi_data['y'] # (N, 4)
    
    print("Executing Spectral Unmixed Mean Pooling (Ignoring vegetated NaNs)...")
    # Spatial mean pooling across only valid pixels (pure soil spectra)
    X_flat = np.nanmean(X, axis=(2, 3)) 
    
    # Fill any edge-case NaNs with column median
    col_mean = np.nanmedian(X_flat, axis=0)
    inds = np.where(np.isnan(X_flat))
    X_flat[inds] = np.take(col_mean, inds[1])
    
    print(f"Flattened Dataset Matrix: {X_flat.shape}")
    
    targets_count = y.shape[1]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    all_r2 = {t: [] for t in range(targets_count)}
    
    print("\n--- PHASE 1: 10-Fold CV Evaluation with Feature Selection ---")
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_flat)):
        print(f"Iterating Fold {fold+1}/10...")
        X_train, X_val = X_flat[train_idx], X_flat[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for t in range(targets_count):
            # Phase 1, Step 1.2 feature selection per target fold
            # RFE with a fast tree to cull down 1290 bands to the top 200!
            selector = RFE(estimator=RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42), n_features_to_select=200, step=100)
            X_t_train = selector.fit_transform(X_train, y_train[:, t])
            X_t_val = selector.transform(X_val)
            
            model = get_stacking_ensemble()
            model.fit(X_t_train, y_train[:, t])
            preds = model.predict(X_t_val)
            
            r2 = r2_score(y_val[:, t], preds)
            all_r2[t].append(r2)
            
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds.")
    
    print("\n------------------------------------------------------------")
    print("PHASE 1 LOG: REPRESENTATION OPTIMIZED METRICS (10-Fold CV)")
    print("------------------------------------------------------------")
    avg_r2_across_targets = []
    for t in range(targets_count):
        mean_r2 = np.mean(all_r2[t])
        avg_r2_across_targets.append(mean_r2)
        print(f"Target {t}: Phase 1 R2 = {mean_r2:.4f}")
        
    print(f"\nGLOBAL PHASE 1 AVERAGE R2: {np.mean(avg_r2_across_targets):.4f}")
    
if __name__ == '__main__':
    train_and_evaluate_phase1()
