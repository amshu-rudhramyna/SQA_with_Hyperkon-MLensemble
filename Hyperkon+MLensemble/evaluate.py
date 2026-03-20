import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.train_phase2 import extract_features
from src.models.ensemble import EnsembleModel

def evaluate_model(data_dir, checkpoint_path):
    print("Extracting full dataset features...")
    X, y = extract_features(data_dir, checkpoint_path)
    
    # Train final model on full dataset
    ensemble = EnsembleModel()
    # Simplified validation split for final metric readout
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print("Training final ensemble on 80% split...")
    ensemble.fit_xgb(X_train, y_train, X_val, y_val)
    ensemble.fit_rf(X_train, y_train)
    ensemble.fit_knn(X_train, y_train)
    
    # For simplicity of visualization, we'll use a standard weighted blend
    weights = {'xgb': 0.6, 'rf': 0.2, 'knn': 0.2}
    print(f"Applying weights: {weights}")
    
    preds = ensemble.predict(X_val, weights=weights)
    
    # Calculate metrics
    metrics = []
    target_names = ['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']
    
    for i, target in enumerate(target_names):
        r2 = r2_score(y_val[:, i], preds[:, i])
        rmse = root_mean_squared_error(y_val[:, i], preds[:, i])
        metrics.append({'Property': target, 'R2': r2, 'RMSE': rmse})
        print(f"{target} - R2: {r2:.4f}, RMSE: {rmse:.4f}")
    
    metrics_df = pd.DataFrame(metrics)
    
    # Plotting
    os.makedirs('results', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, target in enumerate(target_names):
        sns.scatterplot(x=y_val[:, i], y=preds[:, i], ax=axes[i], alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_val[:, i].min(), preds[:, i].min())
        max_val = max(y_val[:, i].max(), preds[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        axes[i].set_title(f'{target} (R²: {metrics_df.iloc[i]["R2"]:.2f})')
        axes[i].set_xlabel('True Values')
        axes[i].set_ylabel('Predicted Values')
        
    plt.tight_layout()
    plt.savefig('results/predictions_vs_truth.png')
    print("Saved plot to results/predictions_vs_truth.png")
    
    # Plot Correlation Matrix
    plt.figure(figsize=(8, 6))
    corr_matrix = pd.DataFrame(preds, columns=target_names).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Predicted Soil Properties')
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png')
    print("Saved plot to results/correlation_matrix.png")
    
if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data', 'raw', 'HYPERVIEW2', 'train')
    
    module_dir = os.path.dirname(os.path.abspath(__file__))
    hyperkon_path = os.path.join(module_dir, 'checkpoints', 'hyperkon_phase1.pth')
    
    evaluate_model(data_dir, hyperkon_path)
