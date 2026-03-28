import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import time
import gc
print("--- [GLOBAL INIT] SoilSpectraNet Phase 4 Pretraining Starting ---", flush=True)

from encoders import SpectralSpatialTransformer, DomainDiscriminator, SpecBPPHead, augment_view_3d, contrastive_loss

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiSensorBPPDset(Dataset):
    """Dataset for Joint Pretraining: Airborne and PRISMA with SpecBPP."""
    def __init__(self, X_air, X_pri):
        self.X_air = X_air.astype(np.float32)
        self.X_pri = X_pri.astype(np.float32)
        self.n_air = len(self.X_air)
        self.n_pri = len(self.X_pri)
        self.length = max(self.n_air, self.n_pri)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_a = torch.from_numpy(self.X_air[idx % self.n_air])
        x_p = torch.from_numpy(self.X_pri[idx % self.n_pri])
        
        # Immediate NaN cleanup
        x_a = torch.nan_to_num(x_a, 0.0)
        x_p = torch.nan_to_num(x_p, 0.0)
        
        # SpecBPP Permutation (50% chance)
        y_bpp_a = 0.0
        if torch.rand(1) > 0.5:
            idx_perm = torch.randperm(x_a.shape[0])
            x_a = x_a[idx_perm]
            y_bpp_a = 1.0
            
        y_bpp_p = 0.0
        if torch.rand(1) > 0.5:
            idx_perm = torch.randperm(x_p.shape[0])
            x_p = x_p[idx_perm]
            y_bpp_p = 1.0
            
        return x_a, torch.tensor([y_bpp_a]), x_p, torch.tensor([y_bpp_p])

def run_transformer_pretraining():
    device = get_device()
    print(f"[Phase 4] Transformer Pretraining (SST + SpecBPP + DANN) | Device: {device}")
    
    data_dir = Path('SoilSpectraNet/data')
    air_data = np.load(data_dir / 'train_hsi_phase1.npz')
    pri_data = np.load(data_dir / 'train_prisma.npz') # Already 2D/3D? 
    # Let's assume PRISMA is already processed to (N, 230, 20, 20) or similar
    X_air = air_data['X']
    X_pri = pri_data['X']
    
    # Standardize PRISMA spatial size to 20x20 if needed
    if X_pri.shape[2] != 20 or X_pri.shape[3] != 20:
        print(f"  Resizing PRISMA from {X_pri.shape[2:]} to 20x20")
        # Simple zero padding or interpolation
        X_p_new = np.zeros((X_pri.shape[0], X_pri.shape[1], 20, 20), dtype=np.float32)
        h, w = X_pri.shape[2], X_pri.shape[3]
        X_p_new[:, :, :h, :w] = X_pri
        X_pri = X_p_new

    dataset = MultiSensorBPPDset(X_air, X_pri)
    # Using 8 batch size and 4 workers for 2x speedup on i9-12900K + 3070 Ti
    loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4) 

    # Models: Separate Spectral Projections, Shared Transformer
    air_sst = SpectralSpatialTransformer(in_bands=1290, embed_dim=128).to(device)
    pri_sst = SpectralSpatialTransformer(in_bands=230, embed_dim=128).to(device)
    
    # Weight Sharing for Transformer & MLP sections
    pri_sst.transformer = air_sst.transformer
    pri_sst.mlp_head = air_sst.mlp_head
    
    dann = DomainDiscriminator(input_dim=128).to(device)
    bpp_head = SpecBPPHead(input_dim=128).to(device)
    
    params = list(air_sst.parameters()) + list(pri_sst.spectral_proj.parameters()) + \
             list(pri_sst.spectral_pool.parameters()) + list(pri_sst.patch_to_embedding.parameters()) + \
             list(dann.parameters()) + list(bpp_head.parameters())
    
    # Doubled learning rate for faster convergence
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    epochs = 10 # 10 epochs with higher LR is equivalent to 15-20 slower ones
    print(f"Starting Joint Training for {epochs} epochs...")
    
    best_loss = float('inf')
    for epoch in range(epochs):
        air_sst.train(); pri_sst.train(); dann.train(); bpp_head.train()
        epoch_loss = 0
        alpha = min(1.0, (epoch + 1) / 10.0) # GRL weight
        
        for i, (x_a, y_b_a, x_p, y_b_p) in enumerate(loader):
            x_a, y_b_a = x_a.to(device), y_b_a.to(device)
            x_p, y_b_p = x_p.to(device), y_b_p.to(device)
            
            # Domain Labels
            y_dom_air = torch.zeros(x_a.size(0), 1).to(device)
            y_dom_pri = torch.ones(x_p.size(0), 1).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            try:
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    # Forward Airborne
                    feat_a = air_sst(x_a)
                    out_bpp_a = bpp_head(feat_a)
                    out_dom_a = dann(feat_a, alpha=alpha)
                    
                    # Forward PRISMA
                    feat_p = pri_sst(x_p)
                    out_bpp_p = bpp_head(feat_p)
                    out_dom_p = dann(feat_p, alpha=alpha)
                    
                    # Contrastive View (Airborne only for simplicity in this batch)
                    view1 = augment_view_3d(x_a)
                    view2 = augment_view_3d(x_a)
                    z1 = air_sst(view1)
                    z2 = air_sst(view2)
                    
                    loss_bpp = F.binary_cross_entropy_with_logits(out_bpp_a, y_b_a) + \
                               F.binary_cross_entropy_with_logits(out_bpp_p, y_b_p)
                    loss_dann = F.binary_cross_entropy_with_logits(out_dom_a, y_dom_air) + \
                                F.binary_cross_entropy_with_logits(out_dom_p, y_dom_pri)
                    loss_cnt = contrastive_loss(z1, z2, temperature=0.1)
                    
                    total_loss = loss_bpp + 0.5 * loss_dann + 0.5 * loss_cnt
                
                if torch.isnan(total_loss):
                    continue
                    
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 0.5)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += total_loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("  OOM. Clearing cache.")
                    torch.cuda.empty_cache()
                    continue
                else: raise e
            
            if (i+1) % 100 == 0:
                print(f"  Batch {i+1}/{len(loader)} | Loss: {total_loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(air_sst.state_dict(), data_dir / 'sst_airborne_v4.pth')
            torch.save(pri_sst.state_dict(), data_dir / 'sst_prisma_v4.pth')
            print("  Models saved.")

    print("Success. Pretraining Complete.")

if __name__ == '__main__':
    run_transformer_pretraining()
