import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import time
import gc
from encoders import SpatialSpectral3DCNN, augment_view_3d, contrastive_loss

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleHSI3DDataset(Dataset):
    """Wrapper for raw 4D HSI patches for contrastive pretraining."""
    def __init__(self, X_np):
        # Convert to float32 and handle NaNs locally
        self.X = X_np.astype(np.float32)
        # Spatial-Spectral 3D CNN expects (B, 1, Bands, H, W)
        # Inside the trainer, we add the channel dim if missing
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        # Clean NaNs per sample to avoid propagation
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=0.0)
        return torch.from_numpy(x)

def run_pretraining_v4():
    device = get_device()
    print(f"[Phase 4] Spatial-Spectral Pretraining on {device}")
    
    data_dir = Path('data')
    hsi_path = data_dir / 'train_hsi_phase1.npz'
    if not hsi_path.exists():
        raise FileNotFoundError("Run phase1_data.py first!")

    print("Loading 4D HSI patches into RAM...")
    data = np.load(hsi_path)
    X_hsi = data['X']  # (B, 1290, 20, 20)
    print(f"Loaded {X_hsi.shape[0]} samples with {X_hsi.shape[1]} bands.")

    dataset = SimpleHSI3DDataset(X_hsi)
    # Reduced batch size for 8GB VRAM with 1290 bands
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    print("Initializing 3D-CNN Encoder...")
    encoder = SpatialSpectral3DCNN(in_bands=X_hsi.shape[1], out_dim=128).to(device)
    
    # Robust optimizer and scheduler
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    epochs = 20
    print(f"Starting Contrastive Pretraining (SimCLR style) for {epochs} epochs...")
    
    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        encoder.train()
        epoch_loss = 0
        valid_batches = 0
        n_batches = 0
        
        for b_x in loader:
            b_x = b_x.to(device, non_blocking=True)
            
            # Global Inf/NaN check for input
            if not torch.isfinite(b_x).all():
                b_x = torch.nan_to_num(b_x, nan=0.0, posinf=1.0, neginf=-1.0)

            noise = 0.005 if epoch < 2 else 0.01
            view1 = augment_view_3d(b_x, noise_std=noise, mask_prob=0.05)
            view2 = augment_view_3d(b_x, noise_std=noise, mask_prob=0.05)
            
            optimizer.zero_grad(set_to_none=True)
            
            try:
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        z1 = encoder(view1)
                        z2 = encoder(view2)
                        loss = contrastive_loss(z1, z2, temperature=0.15) # Relaxed temp
                    
                    if torch.isnan(loss):
                        continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z1 = encoder(view1)
                    z2 = encoder(view2)
                    loss = contrastive_loss(z1, z2, temperature=0.15)
                    if torch.isnan(loss): continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
                    optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
            except RuntimeError as e:
                print(f"  Batch error: {e}")
                continue

            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  Batch {n_batches}/{len(loader)} | Loss: {loss.item():.4f}")
            
            del b_x, view1, view2, z1, z2, loss

        avg_loss = epoch_loss / max(valid_batches, 1)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Valid: {valid_batches}/{len(loader)}")
        
        if valid_batches > 0 and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), data_dir / 'encoder_3d_v4.pth')
            print(f"  Best model saved (Loss: {best_loss:.4f})")

    print(f"Pretraining complete in {time.time() - start_time:.1f}s. Model saved.")
    
    # Cleanup memory
    del encoder, loader, dataset, X_hsi
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == '__main__':
    run_pretraining_v4()
