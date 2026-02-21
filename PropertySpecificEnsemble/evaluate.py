import os
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.train_phase2 import extract_features_and_predictions

def evaluate_property_specific_model(data_dir, checkpoint_path):
    print("Extracting full dataset features focusing on Trace Element Engineering...")
    X_ml, X_cnn_evals, y = extract_features_and_predictions(data_dir, checkpoint_path)
    
    # Static split (80/20) for visual correlation matrix and plots
    split_idx = int(len(X_ml) * 0.8)
    X_train_ml, X_val_ml = X_ml[:split_idx], X_ml[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    X_cnn_train, X_cnn_val = X_cnn_evals[:split_idx], X_cnn_evals[split_idx:]
    
    target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']
    metrics = []
    final_preds = np.zeros_like(y_val)
    
    print("Training decoupled regression endpoints on 80% split...")
    
    for t_idx, target_name in enumerate(target_names):
        y_train_t = y_train[:, t_idx]
        y_val_t = y_val[:, t_idx]
        cnn_val_pred = X_cnn_val[:, t_idx]
        
        # Deploy strict rule-sets
        if target_name in ['B', 'Cu', 'Zn']:
            xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5)
            xgb_model.fit(X_train_ml, y_train_t)
            ml_pred = xgb_model.predict(X_val_ml)
            
            # Boron and Copper scale heavily from SVD/DWT targets in XGBoost
            y_pred = (0.7 * ml_pred) + (0.3 * cnn_val_pred)
            
        elif target_name in ['Fe', 'Mn']:
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf_model.fit(X_train_ml, y_train_t)
            ml_pred = rf_model.predict(X_val_ml)
            
            y_pred = (0.4 * ml_pred) + (0.6 * cnn_val_pred)
            
        elif target_name == 'S':
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4)
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=8)
            xgb_model.fit(X_train_ml, y_train_t)
            rf_model.fit(X_train_ml, y_train_t)
            
            ml_pred = (xgb_model.predict(X_val_ml) + rf_model.predict(X_val_ml)) / 2.0
            y_pred = (0.5 * ml_pred) + (0.5 * cnn_val_pred)
            
        final_preds[:, t_idx] = y_pred
        
        r2 = r2_score(y_val_t, y_pred)
        rmse = root_mean_squared_error(y_val_t, y_pred)
        metrics.append({'Property': target_name, 'R2': r2, 'RMSE': rmse})
        print(f"{target_name} - Final Optimized R2: {r2:.4f}, RMSE: {rmse:.4f}")
        
    metrics_df = pd.DataFrame(metrics)
    
    # Plotting Correlation Matrix
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 6))
    corr_matrix = pd.DataFrame(final_preds, columns=target_names).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title('Property-Specific Model Correlation Matrix')
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png')
    print("Saved decoupled model correlation matrix to results/correlation_matrix.png")
    
if __name__ == "__main__":
    hyperkon_path = 'checkpoints/hyperkon_phase1_trace.pth'
    evaluate_property_specific_model('../data/raw/HYPERVIEW2/train', hyperkon_path)
