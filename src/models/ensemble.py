import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

class EnsembleModel:
    def __init__(self):
        # Best For: P, K, Mg, and pH
        self.xgb_model = xgb.XGBRegressor(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            reg_alpha=0.01, # L1
            reg_lambda=1.0, # L2
            early_stopping_rounds=15,
            random_state=42
        )
        
        # Best For: Organic Matter and Moisture
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            bootstrap=True # Bootstrap sampling as requested
        )
        
        # Best For: Local patterns
        # Normalization applied to KNN input via StandardScaler in a Pipeline
        self.knn_model = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(
                n_neighbors=5,
                weights='distance',
                metric='euclidean'
            ))
        ])

    def fit_xgb(self, X_train, y_train, X_val, y_val, target_names=['P', 'K', 'Mg', 'pH']):
        # XGBoost handles early stopping via eval_set
        print("Training XGBoost...")
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def fit_rf(self, X_train, y_train, sample_weights=None):
        # sample_weights can be inversely proportional to property frequency
        print("Training Random Forest...")
        self.rf_model.fit(X_train, y_train, sample_weight=sample_weights)

    def fit_knn(self, X_train, y_train):
        print("Training KNN...")
        self.knn_model.fit(X_train, y_train)

    def predict(self, X, weights=None):
        """
        Weights is a dictionary with weights for each model.
        e.g., {'xgb': 0.5, 'rf': 0.3, 'knn': 0.2}
        """
        pred_xgb = self.xgb_model.predict(X)
        pred_rf = self.rf_model.predict(X)
        pred_knn = self.knn_model.predict(X)
        
        if weights is not None:
            return (pred_xgb * weights.get('xgb', 1/3) + 
                    pred_rf * weights.get('rf', 1/3) + 
                    pred_knn * weights.get('knn', 1/3))
        
        # Default average
        return (pred_xgb + pred_rf + pred_knn) / 3.0
