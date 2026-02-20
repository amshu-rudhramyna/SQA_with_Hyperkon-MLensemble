import torch
import numpy as np
from sklearn.model_selection import KFold
import warnings
from scipy.optimize import minimize # Alternative to Bayesian Opt for simple weight blending
# from skopt import BayesSearchCV # Better, but minimize is simpler if Bayesian Optimization library isn't available

from src.models.hyperkon import HyperKon
from src.models.ensemble import EnsembleModel
from src.data.loader import HyperviewDataset

warnings.filterwarnings('ignore')

def extract_features(data_dir, model_checkpoint, device='cuda'):
    """ Extract 128-D CNN features + Handcrafted features for the dataset. """
    print("Extracting features using fine-tuned HyperKon...")
    model = HyperKon(num_features=4).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    dataset = HyperviewDataset(data_dir)
    cnn_feats, ml_feats, all_targets = [], [], []

    # Could use DataLoader for batching, doing simple loop for clarity
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            inputs = batch['cnn_input'].unsqueeze(0).to(device) # Add batch dim
            ml_input = batch['ml_features'].numpy()
            targets = batch['targets'].numpy()
            
            emb = model(inputs).squeeze(0).cpu().numpy() # 128-D
            
            cnn_feats.append(emb)
            ml_feats.append(ml_input)
            all_targets.append(targets)
            
    # Combine CNN embeddings with ML features
    X = np.hstack([np.array(cnn_feats), np.array(ml_feats)])
    y = np.array(all_targets)
    
    return X, y

def optimize_ensemble_weights(preds_xgb, preds_rf, preds_knn, y_true):
    """
    Find optimal weights w1, w2, w3 for a single target property using L-BFGS-B (or BO).
    Returns (w1, w2, w3)
    """
    def loss_func(weights):
        w1, w2, w3 = weights
        # Softmax or simple normalization to sum to 1
        w_sum = w1 + w2 + w3
        if w_sum == 0: w_sum = 1e-6
        w1, w2, w3 = w1/w_sum, w2/w_sum, w3/w_sum
        
        y_pred = w1 * preds_xgb + w2 * preds_rf + w3 * preds_knn
        mse = np.mean((y_true - y_pred)**2)
        return mse

    res = minimize(loss_func, x0=[0.33, 0.33, 0.33], bounds=[(0, 1), (0, 1), (0, 1)], method='L-BFGS-B')
    weights = res.x / np.sum(res.x)
    return weights

def train_phase2(data_dir, hyperkon_ckpt):
    X, y = extract_features(data_dir, hyperkon_ckpt)
    
    print(f"Extracted feature matrix shape: {X.shape}")
    
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ensemble = EnsembleModel()
    
    property_weights = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # In a real scenario, we'd train the ensemble for each property
        # For simplicity, assuming y_train is single target or ensemble handles multi-output.
        # XGBoost and RF can handle multi-output, but weights are often property-specific.
        
        # Fit models
        ensemble.fit_xgb(X_train, y_train, X_val, y_val)
        ensemble.fit_rf(X_train, y_train)
        ensemble.fit_knn(X_train, y_train)
        
        # We need predictions on Validation to optimize weights
        pred_xgb = ensemble.xgb_model.predict(X_val)
        pred_rf = ensemble.rf_model.predict(X_val)
        pred_knn = ensemble.knn_model.predict(X_val)
        
        # If multi-target, optimize per target (e.g. 4 targets)
        num_targets = y_train.shape[1] if len(y_train.shape) > 1 else 1
        fold_weights = []
        for t in range(num_targets):
            best_w = optimize_ensemble_weights(
                pred_xgb[:, t] if num_targets > 1 else pred_xgb,
                pred_rf[:, t] if num_targets > 1 else pred_rf,
                pred_knn[:, t] if num_targets > 1 else pred_knn,
                y_val[:, t] if num_targets > 1 else y_val
            )
            fold_weights.append(best_w)
            print(f"Target {t} Optimal Weights: XGB={best_w[0]:.2f}, RF={best_w[1]:.2f}, KNN={best_w[2]:.2f}")
            
        property_weights.append(fold_weights)

    print("\nPhase 2 Complete. Ensemble weights optimized.")
    return property_weights

if __name__ == "__main__":
    # train_phase2('data/raw/HYPERVIEW2/train', 'checkpoints/hyperkon_phase1.pth')
    pass
