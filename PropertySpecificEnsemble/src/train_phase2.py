import os
import sys
import torch
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error

# Ensure root directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hyperkon import HyperKon
from src.data.loader import HyperviewDataset

def extract_features_and_predictions(data_dir, model_checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("Extracting features using Phase 1 HyperKon...")
    # B, Cu, Zn, Fe, S, Mn
    model = HyperKon(num_features=6).to(device)
    
    # Check if checkpoint exists, otherwise warn (useful for dry runs before full Phase 1 scaling)
    if os.path.exists(model_checkpoint):
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    else:
        print(f"Warning: {model_checkpoint} not found. Running with untrained Phase 1 weights for structural validation.")
        
    model.eval()
    dataset = HyperviewDataset(data_dir)
    
    cnn_feats, ml_feats, cnn_preds, all_targets = [], [], [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            inputs = batch['cnn_input'].unsqueeze(0).to(device)
            
            # Use AMP during evaluation extraction as well
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=(device=='cuda')):
                # Direct CNN prediction mapped directly out of the head (6 target nodes)
                preds = model(inputs).squeeze(0).cpu().numpy()
                # To get the 128-D embedding, we need to bypass the final `model.fc` layer
                # HyperKon structurally outputs the embedding from the SE blocks right before `self.fc`
                # We can access `model.fc` inputs by feeding forward manually, but for brevity, 
                # we'll capture the representation directly if modified, or just use the ML transforms + cnn_preds.
                # Actually, the user instructed to blend CNN predictions with ML predictions outright!
            
            ml_input = batch['ml_features'].numpy()
            targets = batch['targets'].numpy()
            
            cnn_preds.append(preds)
            ml_feats.append(ml_input)
            all_targets.append(targets)
            
    # Compile
    X_ml = np.array(ml_feats)
    X_cnn_evals = np.array(cnn_preds)
    y = np.array(all_targets)
    
    return X_ml, X_cnn_evals, y

def train_phase2(data_dir, hyperkon_ckpt):
    X_ml, X_cnn_evals, y = extract_features_and_predictions(data_dir, hyperkon_ckpt)
    
    print(f"Feature matrix shapes -> ML: {X_ml.shape} | CNN Outputs: {X_cnn_evals.shape}")
    
    targets = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store metrics over 5 folds
    target_metrics = {t: {'R2': [], 'RMSE': []} for t in targets}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_ml)):
        print(f"\n--- Fold {fold+1} ---")
        
        for t_idx, target_name in enumerate(targets):
            y_train, y_val = y[train_idx, t_idx], y[val_idx, t_idx]
            X_train_ml, X_val_ml = X_ml[train_idx], X_ml[val_idx]
            
            # The CNN predictions are static baseline truths extracted from the Phase 1 model
            cnn_pred_val = X_cnn_evals[val_idx, t_idx]
            final_val_pred = np.zeros_like(cnn_pred_val)
            
            # --- TIER 3: PROPERTY SPECIFIC ENSEMBLE ROUTING ---
            
            if target_name in ['B', 'Cu', 'Zn']:
                # Heavy XGBoost reliance over SVD/DWT engineered ML targets (0.7 ML : 0.3 CNN)
                xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, objective='reg:squarederror')
                xgb_model.fit(X_train_ml, y_train)
                ml_pred = xgb_model.predict(X_val_ml)
                
                final_val_pred = (0.7 * ml_pred) + (0.3 * cnn_pred_val)
                
            elif target_name in ['Fe', 'Mn']:
                # Heavy Deep Learning shape reliance with minor RF clustering (0.6 CNN : 0.4 ML)
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                rf_model.fit(X_train_ml, y_train)
                ml_pred = rf_model.predict(X_val_ml)
                
                final_val_pred = (0.4 * ml_pred) + (0.6 * cnn_pred_val)
                
            elif target_name == 'S':
                # Balanced XGBoost/Random Forest dual-learning mapped neutrally to CNN
                xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4)
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=8)
                
                xgb_model.fit(X_train_ml, y_train)
                rf_model.fit(X_train_ml, y_train)
                
                ml_pred = (xgb_model.predict(X_val_ml) + rf_model.predict(X_val_ml)) / 2.0
                
                final_val_pred = (0.5 * ml_pred) + (0.5 * cnn_pred_val)
                
            # Score
            r2 = r2_score(y_val, final_val_pred)
            rmse = root_mean_squared_error(y_val, final_val_pred)
            
            target_metrics[target_name]['R2'].append(r2)
            target_metrics[target_name]['RMSE'].append(rmse)
            
            print(f"  {target_name}: R2 = {r2:.4f} | RMSE = {rmse:.4f}")
            
    print("\n=== Phase 2 Cross-Validation Results ===")
    for t in targets:
        avg_r2 = np.mean(target_metrics[t]['R2'])
        avg_rmse = np.mean(target_metrics[t]['RMSE'])
        print(f"{t}: Final R2: {avg_r2:.4f} | Final RMSE: {avg_rmse:.4f}")

if __name__ == "__main__":
    hyperkon_path = 'checkpoints/hyperkon_phase1_trace.pth'
    train_phase2('../data/raw/HYPERVIEW2/train', hyperkon_path)
