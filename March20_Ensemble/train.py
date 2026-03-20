import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import optuna
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 1. DATA LOADING
def load_data(path):
    data = np.load(path)
    X = data['X']
    y = data['y']
    return X, y

# 2. PREPROCESSING
def smooth(X):
    return savgol_filter(X, window_length=11, polyorder=2, axis=1)

# 3. FEATURE ENGINEERING
def spectral_derivatives(X):
    d1 = np.gradient(X, axis=1)
    d2 = np.gradient(d1, axis=1)
    return np.concatenate([X, d1, d2], axis=1)

# 4. FEATURE SELECTION (CRITICAL)
def select_features(X, y):
    # Using 50 estimators to speed up RFE on potentially large features
    selector = RFE(RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1), n_features_to_select=60, step=0.1)
    X_selected = selector.fit_transform(X, y[:, 0])
    return X_selected, selector

# 5. DATA AUGMENTATION
def augment(X):
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise

# 6. DEEP FEATURE EXTRACTION (CNN)
class SpectralCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)

def train_cnn(model, X_train, y_train, epochs=20, lr=1e-3):
    """Basic training loop to learn representations from the target."""
    model.train()
    head = nn.Linear(64, y_train.shape[1])
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Training SpectralCNN...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            features = model(bx)
            preds = head(features)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

# 7. EXTRACT CNN FEATURES
def extract_features(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        features = model(X_tensor).numpy()
    return features

def main():
    print("Loading data...")
    train_path = Path('data/train.npz')
    test_path = Path('data/test.npz')
    
    if not train_path.exists():
        print("Data not found. Please run prepare_data.py first.")
        return
        
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    print("Applying Savitzky-Golay smoothing...")
    X_train = smooth(X_train)
    X_test = smooth(X_test)

    print("Adding spectral derivatives...")
    X_train = spectral_derivatives(X_train)
    X_test = spectral_derivatives(X_test)

    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Performing feature selection (RFE)...")
    X_train, selector = select_features(X_train, y_train)
    X_test = selector.transform(X_test)

    print("Augmenting training data with spectral noise...")
    X_train_aug = augment(X_train)
    X_train_combined = np.vstack((X_train, X_train_aug))
    y_train_combined = np.vstack((y_train, y_train))
    
    print("Initializing and training SpectralCNN...")
    cnn = SpectralCNN(X_train_combined.shape[1])
    train_cnn(cnn, X_train_combined, y_train_combined, epochs=200)

    print("Extracting CNN features...")
    cnn_features_train = extract_features(cnn, X_train_combined)
    cnn_features_test = extract_features(cnn, X_test)
    
    print("Concatenating ML features with CNN embeddings...")
    features_train = np.hstack((X_train_combined, cnn_features_train))
    features_test = np.hstack((X_test, cnn_features_test))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("Running Optuna for Random Forest hyperparameter tuning...")
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        max_depth = trial.suggest_int("max_depth", 5, 20)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        X_t, X_v, y_t, y_v = train_test_split(features_train, y_train_combined[:, 0], test_size=0.2, random_state=42)
        model.fit(X_t, y_t)
        return model.score(X_v, y_v)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    best_rf_params = study.best_params
    print(f"Best RF Params: {best_rf_params}")
    
    print("Building Stacking Ensemble...")
    estimators = [
        ('rf', RandomForestRegressor(
            n_estimators=best_rf_params['n_estimators'], 
            max_depth=best_rf_params['max_depth'],
            random_state=42, n_jobs=-1
        )),
        ('svr', SVR()),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)) # Reduced estimators for speed
    ]

    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge()
    )

    print("Wrapping in MultiOutputRegressor and fitting...")
    model = MultiOutputRegressor(stack_model)
    model.fit(features_train, y_train_combined)
    
    print("Evaluating on test set...")
    pred = model.predict(features_test)
    
    overall_r2 = r2_score(y_test, pred)
    overall_rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"\n--- RESULTS ---")
    print(f"Overall R2: {overall_r2:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    
    targets = ['B', 'Cu', 'Zn', 'Fe']
    for i in range(pred.shape[1]):
        r2 = r2_score(y_test[:, i], pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], pred[:, i]))
        t_name = targets[i] if i < len(targets) else f"Target_{i}"
        print(f" - {t_name}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

if __name__ == '__main__':
    main()
