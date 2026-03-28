import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from encoders import SpatialSpectral3DCNN, SpectralCNN, pretrain_encoder
from ensemble import TargetSpecificLocalStacking

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(data_dir):
    hsi_data = np.load(data_dir / 'train_hsi.npz')
    msi_data = np.load(data_dir / 'train_msi.npz')
    
    X_hsi = hsi_data['X']
    X_msi = msi_data['X']
    y = hsi_data['y']
    
    return X_hsi, X_msi, y

def extract_features(encoder, X_data, device, batch_size=64):
    """
    Pass raw tensors through the pretrained GPU network to extract numerical embeddings.
    """
    encoder.eval()
    encoder.to(device)
    
    dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            with torch.amp.autocast('cuda'):
                feats = encoder(batch_x)
            features.append(feats.cpu().numpy())
            
    return np.concatenate(features, axis=0)

if __name__ == '__main__':
    data_dir = Path('data')
    X_hsi, X_msi, y = load_data(data_dir)
    print(f"Data shapes loaded: HSI={X_hsi.shape}, MSI={X_msi.shape}, Targets={y.shape}")
    
    indices = np.arange(len(X_hsi))
    # 80/20 train test split to validate local modeling
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    device = get_device()
    print(f"GPU Engine Status: {device}")
    
    hsi_bands = X_hsi.shape[1]
    msi_bands = X_msi.shape[1]
    
    # ---------------------------------------------------------
    # PART 1: UN-SUPERVISED DEEP FEATURE LEARNING (RTX 3070 Ti)
    # ---------------------------------------------------------
    print("\n[PART 1/3] Pretraining Spatial-Spectral 3D CNN (HSI)...")
    hsi_loader_train = DataLoader(TensorDataset(torch.tensor(X_hsi[train_idx], dtype=torch.float32)), 
                                  batch_size=32, shuffle=True, pin_memory=True)
                                  
    # Use 128 dimension dense embedding
    hsi_encoder = SpatialSpectral3DCNN(in_bands=hsi_bands, out_dim=128)
    hsi_encoder = pretrain_encoder(hsi_encoder, hsi_loader_train, device, epochs=15)
    
    print("\n[PART 1/3] Pretraining 2D Spectral CNN (MSI)...")
    msi_loader_train = DataLoader(TensorDataset(torch.tensor(X_msi[train_idx], dtype=torch.float32)), 
                                  batch_size=32, shuffle=True, pin_memory=True)
                                  
    # Use 32 dimension dense embedding for shallow satellite data
    msi_encoder = SpectralCNN(in_bands=msi_bands, out_dim=32)
    # We use lower epochs for MSI because it's significantly smaller dimensional space
    msi_encoder = pretrain_encoder(msi_encoder, msi_loader_train, device, epochs=10)
    
    # ---------------------------------------------------------
    # PART 2: GPU FEATURE EXTRACTION & LATE FUSION
    # ---------------------------------------------------------
    print("\n[PART 2/3] Extracting latent vector embeddings across all instances...")
    Z_hsi_train = extract_features(hsi_encoder, X_hsi[train_idx], device)
    Z_hsi_test = extract_features(hsi_encoder, X_hsi[test_idx], device)
    
    Z_msi_train = extract_features(msi_encoder, X_msi[train_idx], device)
    Z_msi_test = extract_features(msi_encoder, X_msi[test_idx], device)
    
    # Concat
    Z_train = np.concatenate([Z_hsi_train, Z_msi_train], axis=1) # Shape: (1500, 160)
    Z_test = np.concatenate([Z_hsi_test, Z_msi_test], axis=1)    # Shape: (376, 160)
    
    print(f"Fused Training Matrix: {Z_train.shape}")
    
    # ---------------------------------------------------------
    # PART 3: CPU MULTI-THREADED LOCAL KNN STACKING (i9-12900K)
    # ---------------------------------------------------------
    print("\n[PART 3/3] Triggering CPU Neighborhood Stacking Regression...")
    targets_count = y.shape[1]
    
    # Pass dense representation matrix into our Stacking Algorithm
    # Neighborhood threshold tuned to K=150 to reduce overfitting vs 200
    local_ensemble = TargetSpecificLocalStacking(k_neighbors=150, targets=targets_count)
    local_ensemble.fit(Z_train, y[train_idx])
    
    y_pred = local_ensemble.predict(Z_test)
    
    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------
    print("\n=== HYBRID PIPELINE FINAL RESULTS ===")
    r2_scores = []
    for t in range(targets_count):
        r2 = r2_score(y[test_idx][:, t], y_pred[:, t])
        print(f"Target {t} Final R2: {r2:.4f}")
        r2_scores.append(r2)
        
    avg_r2 = np.mean(r2_scores)
    print(f"Hybrid Average R2: {avg_r2:.4f}")
    
    if avg_r2 > 0.85:
        print("SUCCESS: Surpassed the 0.85 evaluation threshold!")
    else:
        print("Note: Approaching limit constraint of the 1,500 sample dataset bounds.")
