import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from encoders import SpatialSpectral3DCNN, SpectralCNN, CrossAttentionFusion, pretrain_encoder

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(data_dir):
    hsi_data = np.load(data_dir / 'train_hsi.npz')
    msi_data = np.load(data_dir / 'train_msi.npz')
    
    X_hsi = hsi_data['X']
    X_msi = msi_data['X']
    y = hsi_data['y']
    
    return X_hsi, X_msi, y

class SharedMultiHeadRegressor(nn.Module):
    """
    Approach 2: Multi-head Shared Neural Network.
    Takes fused Cross-Attention features and branches into individual dense layers for targets.
    """
    def __init__(self, in_features, num_targets):
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Target specific heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_targets)
        ])
        
    def forward(self, x):
        shared_feat = self.trunk(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(shared_feat))
        # outputs shape: (num_targets, B, 1). Return (B, num_targets)
        return torch.cat(outputs, dim=1)

class SoilSpectraNet(nn.Module):
    """
    The full Unified End-to-End PyTorch Network (Approach 2)
    """
    def __init__(self, hsi_bands, msi_bands, num_targets):
        super().__init__()
        self.hsi_encoder = SpatialSpectral3DCNN(in_bands=hsi_bands, out_dim=128)
        self.msi_encoder = SpectralCNN(in_bands=msi_bands, out_dim=64)
        
        self.fusion = CrossAttentionFusion(hsi_dim=128, msi_dim=64, embed_dim=128, num_heads=4)
        
        self.multihead_regressor = SharedMultiHeadRegressor(in_features=128, num_targets=num_targets)
        
    def forward(self, hsi, msi):
        hsi_feat = self.hsi_encoder(hsi)
        msi_feat = self.msi_encoder(msi)
        
        fused = self.fusion(hsi_feat, msi_feat)
        preds = self.multihead_regressor(fused)
        return preds

def train_end_to_end(model, train_loader, val_loader, device, epochs=50):
    # Approach 2: Strong regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    model.to(device)
    
    best_val_r2 = -float('inf')
    
    print("\n--- PHASE 3: Training Unified Multimodal Network ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for b_hsi, b_msi, b_y in train_loader:
            b_hsi, b_msi, b_y = b_hsi.to(device), b_msi.to(device), b_y.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                preds = model(b_hsi, b_msi)
                loss = criterion(preds, b_y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for b_hsi, b_msi, b_y in val_loader:
                b_hsi, b_msi = b_hsi.to(device), b_msi.to(device)
                with torch.amp.autocast('cuda'):
                    preds = model(b_hsi, b_msi)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(b_y.numpy())
                
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        r2_scores = [r2_score(val_targets[:, t], val_preds[:, t]) for t in range(val_targets.shape[1])]
        avg_r2 = np.mean(r2_scores)
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss/len(train_loader):.4f} | Val R2: {avg_r2:.4f}")
            
        if avg_r2 > best_val_r2:
            best_val_r2 = avg_r2
            torch.save(model.state_dict(), "best_soilspectranet.pth")
            
    print(f"\nTraining Complete. Best Validation Average R2: {best_val_r2:.4f}")
    return model

if __name__ == '__main__':
    data_dir = Path('data')
    X_hsi, X_msi, y = load_data(data_dir)
    print(f"Shapes loaded: HSI={X_hsi.shape}, MSI={X_msi.shape}, Y={y.shape}")
    
    indices = np.arange(len(X_hsi))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Pretraining (just HSI to be fast, MSI is shallow)
    print("\n--- PHASE 1: Self-Supervised Pretraining (HSI 3D CNN) ---")
    hsi_bands = X_hsi.shape[1]
    hsi_loader_train = DataLoader(TensorDataset(torch.tensor(X_hsi[train_idx], dtype=torch.float32)), 
                                  batch_size=32, shuffle=True, pin_memory=True)
    
    pretrained_hsi = SpatialSpectral3DCNN(in_bands=hsi_bands, out_dim=128)
    pretrained_hsi = pretrain_encoder(pretrained_hsi, hsi_loader_train, device, epochs=15)
    
    print("\n--- PHASE 2: Initializing Full Model ---")
    msi_bands = X_msi.shape[1]
    num_targets = y.shape[1]
    
    net = SoilSpectraNet(hsi_bands=hsi_bands, msi_bands=msi_bands, num_targets=num_targets)
    # Inject pretrained weights
    net.hsi_encoder.load_state_dict(pretrained_hsi.state_dict())
    
    train_dataset = TensorDataset(
        torch.tensor(X_hsi[train_idx], dtype=torch.float32),
        torch.tensor(X_msi[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_hsi[test_idx], dtype=torch.float32),
        torch.tensor(X_msi[test_idx], dtype=torch.float32),
        torch.tensor(y[test_idx], dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    train_end_to_end(net, train_loader, test_loader, device, epochs=50)

    print("\nFinal Model Evaluation...")
    net.load_state_dict(torch.load("best_soilspectranet.pth", weights_only=True))
    net.eval()
    net.to(device)
    
    all_preds = []
    with torch.no_grad():
        for b_hsi, b_msi, _ in test_loader:
            p = net(b_hsi.to(device), b_msi.to(device))
            all_preds.append(p.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    
    print("\n--- RESULTS PHASE 3 ---")
    final_r2s = []
    for t in range(num_targets):
        r2 = r2_score(y[test_idx][:, t], all_preds[:, t])
        print(f"Target {t} Final R2: {r2:.4f}")
        final_r2s.append(r2)
        
    print(f"Overall Approach 2 Performance Average R2: {np.mean(final_r2s):.4f}")
