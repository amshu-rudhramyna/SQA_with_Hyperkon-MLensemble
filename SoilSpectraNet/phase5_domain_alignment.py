import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pathlib import Path
import time
import gc
from encoders import SpatialSpectral3DCNN, SpectralCNN, DomainDiscriminator, SpecBPPHead, augment_view_3d

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DomainDataset(Dataset):
    """Combines Pruned Airborne and PRISMA data for sensor-agnostic pretraining."""
    def __init__(self, X_airborne, X_prisma):
        self.X_air = X_airborne.astype(np.float32)
        self.X_pri = X_prisma.astype(np.float32)
        self.n_air = len(self.X_air)
        self.n_pri = len(self.X_pri)
        self.length = max(self.n_air, self.n_pri)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Sample from both (simple wrapping for minority class)
        air_idx = idx % self.n_air
        pri_idx = idx % self.n_pri
        
        x_air = torch.from_numpy(self.X_air[air_idx])
        x_pri = torch.from_numpy(self.X_pri[pri_idx])
        
        # Immediate cleanup and robust clamping for numerical safety
        if not torch.isfinite(x_air).all():
            x_air = torch.nan_to_num(x_air, nan=0.0)
        x_air = torch.clamp(x_air, min=-10.0, max=10.0)

        if not torch.isfinite(x_pri).all():
            x_pri = torch.nan_to_num(x_pri, nan=0.0)
        x_pri = torch.clamp(x_pri, min=-10.0, max=10.0)
            
        # SpecBPP Permutation Logic (Binary Task)
        # 50% chance to shuffle bands for each sensor
        y_bpp_air = 0.0
        if torch.rand(1) > 0.5:
            indices = torch.randperm(x_air.shape[0])
            x_air = x_air[indices]
            y_bpp_air = 1.0
            
        y_bpp_pri = 0.0
        if torch.rand(1) > 0.5:
            indices = torch.randperm(x_pri.shape[0])
            x_pri = x_pri[indices]
            y_bpp_pri = 1.0
            
        return {
            'x_air': x_air, 'y_bpp_air': torch.tensor([y_bpp_air]),
            'x_pri': x_pri, 'y_bpp_pri': torch.tensor([y_bpp_pri])
        }

def run_domain_pretraining():
    device = get_device()
    print(f"[Phase 3] Domain Alignment & SpecBPP Pretraining on {device}")
    
    data_dir = Path('data')
    air_path = data_dir / 'train_hsi_pruned.npz'
    pri_path = data_dir / 'train_prisma.npz'
    
    if not air_path.exists() or not pri_path.exists():
        raise FileNotFoundError("Run phase1_2_pruning.py first!")

    print("Loading Pruned Airborne and PRISMA patches...")
    d_air = np.load(air_path)
    d_pri = np.load(pri_path)
    
    dataset = DomainDataset(d_air['X'], d_pri['X'])
    loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True)

    # Models
    air_enc = SpatialSpectral3DCNN(in_bands=512, out_dim=128).to(device)
    pri_enc = SpectralCNN(in_bands=230, out_dim=128).to(device)
    dann = DomainDiscriminator(input_dim=128).to(device)
    bpp_head = SpecBPPHead(input_dim=128).to(device)
    
    params = list(air_enc.parameters()) + list(pri_enc.parameters()) + \
             list(dann.parameters()) + list(bpp_head.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    epochs = 20
    print(f"Starting Joint Pretraining (SpecBPP + DANN) for {epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        air_enc.train(); pri_enc.train(); dann.train(); bpp_head.train()
        epoch_loss = 0
        
        # Hyperparameter alpha for GRL increases with progress
        alpha = min(1.0, (epoch + 1) / 10.0)
        
        for batch in loader:
            x_a = batch['x_air'].to(device, non_blocking=True)
            y_b_a = batch['y_bpp_air'].to(device, non_blocking=True)
            x_p = batch['x_pri'].to(device, non_blocking=True)
            y_b_p = batch['y_bpp_pri'].to(device, non_blocking=True)
            
            # Labels for Domain Discriminator (Air=0, Pri=1)
            y_dom_air = torch.zeros(x_a.size(0), 1).to(device)
            y_dom_pri = torch.ones(x_p.size(0), 1).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    # Forward Airborne
                    feat_a = air_enc(x_a)
                    out_bpp_a = bpp_head(feat_a)
                    out_dom_a = dann(feat_a, alpha=alpha)
                    
                    # Forward PRISMA
                    feat_p = pri_enc(x_p)
                    out_bpp_p = bpp_head(feat_p)
                    out_dom_p = dann(feat_p, alpha=alpha)
                    
                    # Losses (using with_logits for autocast safety)
                    loss_bpp = F.binary_cross_entropy_with_logits(out_bpp_a, y_b_a) + \
                               F.binary_cross_entropy_with_logits(out_bpp_p, y_b_p)
                    loss_dann = F.binary_cross_entropy_with_logits(out_dom_a, y_dom_air) + \
                                F.binary_cross_entropy_with_logits(out_dom_p, y_dom_pri)
                    
                    total_loss = loss_bpp + 0.5 * loss_dann
                    
                scaler.scale(total_loss).backward()
                # Grad Clipping for health
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            epoch_loss += total_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Joint Loss: {epoch_loss/len(loader):.4f} | Alpha: {alpha:.2f}")

    print("Success. Saving Aligned Multi-Sensor Encoders...")
    torch.save(air_enc.state_dict(), data_dir / 'encoder_airborne_v3.pth')
    torch.save(pri_enc.state_dict(), data_dir / 'encoder_prisma_v3.pth')
    print(f"Pretraining complete in {time.time() - start_time:.1f}s.")

if __name__ == '__main__':
    run_domain_pretraining()
