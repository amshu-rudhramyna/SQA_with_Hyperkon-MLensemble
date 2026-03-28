import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import joblib

def get_base_models():
    """
    Component 5: Ensemble (Stacking) approach.
    Base models: RF, XGB, SVR
    """
    return [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42)),
        ('svr', SVR(C=1.0, epsilon=0.1))
    ]

def get_stacking_ensemble():
    """
    Component 5: Meta-learner is Ridge Regression.
    """
    level0 = get_base_models()
    level1 = Ridge(alpha=1.0)
    # StackingRegressor uses K-Fold cross val for training meta-model
    model = StackingRegressor(estimators=level0, final_estimator=level1, n_jobs=1)
    return model

class TargetSpecificLocalStacking:
    """
    Component 4 (Local Spectral Modeling): KNN-based Local Training
    Component 6 (Target-Specific Modeling): Separate models per target
    """
    def __init__(self, k_neighbors=300, targets=4):
        self.k_neighbors = k_neighbors
        self.targets = targets
        self.knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', n_jobs=-1)
        self.X_train = None
        self.Y_train = None
        
    def fit(self, X, y):
        """
        X: features (N, D)
        y: targets (N, targets)
        """
        self.X_train = X
        self.Y_train = y
        self.knn.fit(X)
        return self

    def predict(self, X_test):
        """
        For each test sample, find K nearest neighbors from train set, 
        fit separate stacking models for each target, and predict.
        """
        N_test = len(X_test)
        preds = np.zeros((N_test, self.targets), dtype=np.float32)
        
        # Parallel processing for test samples could be added here
        # but for simplicity and Thread safety inside StackingRegressor it's done sequentially
        count = 0
        distances, indices = self.knn.kneighbors(X_test)
        
        for i in range(N_test):
            neighbor_idx = indices[i]
            x_local = self.X_train[neighbor_idx]
            y_local = self.Y_train[neighbor_idx]
            
            # Predict each target separately
            for t in range(self.targets):
                model = get_stacking_ensemble()
                y_t = y_local[:, t]
                model.fit(x_local, y_t)
                pred_val = model.predict(X_test[i:i+1])
                preds[i, t] = pred_val[0]
            
            count += 1
            if count % 20 == 0:
                print(f"Predicted {count}/{N_test} local models...")
                
        return preds
