import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
import optuna
from pathlib import Path
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# 1. ADVANCED PREPROCESSING
def preprocess(X):
    print("Applying Savitzky-Golay smoothing...")
    X_smooth = savgol_filter(X, 11, 2, axis=1)

    print("Applying Standard Normal Variate (SNV)...")
    X_snv = (X_smooth - X_smooth.mean(axis=1, keepdims=True)) / (X_smooth.std(axis=1, keepdims=True) + 1e-8)

    print("Calculating Spectral Derivatives...")
    d1 = np.gradient(X_snv, axis=1)
    d2 = np.gradient(d1, axis=1)

    return np.concatenate([X_snv, d1, d2], axis=1)

# 2 & 4. TRANSFORMER & PRETRAINING
class SpectralTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.encoder(x)

def augment_batch(X_batch):
    noise = torch.randn_like(X_batch) * 0.01
    mask = (torch.rand_like(X_batch) > 0.1).float()
    
    # Spectral shift 
    shift = torch.randint(1, 5, (1,)).item()
    X3 = torch.roll(X_batch, shifts=shift, dims=1)
    
    return X_batch + noise, X_batch * mask

def contrastive_loss(z1, z2):
    return ((z1 - z2)**2).mean()

def pretrain_transformer(model, X_train, epochs=150, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_tensor = X_tensor.to(device)
    
    print(f"Pretraining Transformer on {device} (Full Batch)...")
    for epoch in range(epochs):
        x1, x2 = augment_batch(X_tensor)
        
        z1 = model(x1)
        z2 = model(x2)
        loss = contrastive_loss(z1, z2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Pretrain Epoch {epoch+1}/{epochs}, SSL Loss: {loss.item():.4f}")
    return model

def extract_transformer_features(model, X):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_tensor).cpu().numpy()

# 3, 6, 7, 8. LOCAL TARGET-WISE STACKING WITH JOB-LIB
def train_target_models_locally(X_train, y_train, X_test, target_idx, tune=True):
    y_t = y_train[:, target_idx]
    print(f"\n--- Processing Target {target_idx} ---")
    
    best_params = {'max_depth': 12, 'learning_rate': 0.05, 'n_estimators': 500}
    
    if tune:
        print(f"Optuna Tuning global subsets for localized Base Models (Target {target_idx})...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            lr = trial.suggest_float("learning_rate", 0.01, 0.3)
            md = trial.suggest_int("max_depth", 5, 20)
            n_est = trial.suggest_int("n_estimators", 200, 1000)
            
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            model = LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=42, verbose=-1, n_jobs=-1)
            
            scores = []
            for tr_idx, va_idx in kf.split(X_train):
                model.fit(X_train[tr_idx], y_t[tr_idx])
                preds = model.predict(X_train[va_idx])
                scores.append(r2_score(y_t[va_idx], preds))
            return np.mean(scores)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        print(f"Best Params Target {target_idx}: {best_params}")
    
    # Base Stack definition
    def get_stack_model():
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=best_params['max_depth'], random_state=42, n_jobs=-1)),
            ('xgb', XGBRegressor(n_estimators=100, learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'], random_state=42, n_jobs=-1)),
            ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'], random_state=42, verbose=-1, n_jobs=-1))
        ]
        meta = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
        # Using Step 9 KFold split inside Stacker!
        return StackingRegressor(estimators=estimators, final_estimator=meta, cv=KFold(n_splits=3, shuffle=True, random_state=42))

    print("Training decoupled Target Stack on ALL training features natively...")
    global_model = get_stack_model()
    global_model.fit(X_train, y_t)
    
    print("Executing batch inference across test set...")
    preds = global_model.predict(X_test)
    
    return np.array(preds)


def main():
    print("Loading data...")
    train_path = Path('data/train.npz')
    test_path = Path('data/test.npz')
    
    if not train_path.exists():
        print("Data not found. Please run prepare_data.py first.")
        return
        
    data_train = np.load(train_path)
    data_test = np.load(test_path)
    X_train_raw = data_train['X']
    y_train = data_train['y']
    X_test_raw = data_test['X']
    y_test = data_test['y']
    print(f"Train shapes: X={X_train_raw.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test_raw.shape}, y={y_test.shape}")
    
    print("\n--- STEP 1: PREPROCESSING ---")
    X_train_pre = preprocess(X_train_raw)
    X_test_pre = preprocess(X_test_raw)
    
    print("\n--- STEP 2 & 4: TRANSFORMER PRETRAINING ---")
    transformer = SpectralTransformer(dim=150)
    pretrain_transformer(transformer, X_train_raw, epochs=150)
    
    print("\n--- STEP 5: FEATURE FUSION ---")
    cnn_features_train = extract_transformer_features(transformer, X_train_raw)
    cnn_features_test = extract_transformer_features(transformer, X_test_raw)
    
    X_train_final = np.concatenate([cnn_features_train, X_train_pre], axis=1)
    X_test_final = np.concatenate([cnn_features_test, X_test_pre], axis=1)
    print(f"Final feature arrays constructed. Shape: {X_train_final.shape}")
    
    print("\n--- STEP 3, 6, 7, 8: TARGET-WISE LOCAL MODELING ---")
    predictions = np.zeros_like(y_test)
    
    targets = ['B', 'Cu', 'Zn', 'Fe']
    for i in range(y_train.shape[1]):
        preds_i = train_target_models_locally(X_train_final, y_train, X_test_final, target_idx=i, tune=True)
        predictions[:, i] = preds_i
        r2 = r2_score(y_test[:, i], preds_i)
        print(f">>> Target {targets[i]} Local Stack R2: {r2:.4f} <<<")

    print("\n====================")
    print("   FINAL METRICS     ")
    print("====================")
    
    overall_r2 = r2_score(y_test, predictions)
    overall_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"OVERALL R2: {overall_r2:.4f}")
    print(f"OVERALL RMSE: {overall_rmse:.4f}")
    
    for i in range(predictions.shape[1]):
        r2 = r2_score(y_test[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))
        print(f" - {targets[i]}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

if __name__ == '__main__':
    main()
