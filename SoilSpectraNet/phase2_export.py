import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import gc
import time
import warnings
warnings.filterwarnings('ignore')

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpectralEncoder1D(nn.Module):
    def __init__(self, in_bands=1290, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
        )
        self.block1 = self._make_res_block(64, 128, kernel_size=7, pool=4)
        self.block2 = self._make_res_block(128, 256, kernel_size=5, pool=4)
        self.block3 = self._make_res_block(256, 256, kernel_size=3, pool=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def _make_res_block(self, in_ch, out_ch, kernel_size, pool):
        return nn.ModuleDict({
            'conv1': nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            'bn1': nn.BatchNorm1d(out_ch),
            'conv2': nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            'bn2': nn.BatchNorm1d(out_ch),
            'skip': nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity(),
            'pool': nn.MaxPool1d(pool),
        })

    def _forward_block(self, block, x):
        residual = block['skip'](x)
        x = F.gelu(block['bn1'](block['conv1'](x)))
        x = block['bn2'](block['conv2'](x))
        x = F.gelu(x + residual)
        x = block['pool'](x)
        return x

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self._forward_block(self.block1, x)
        x = self._forward_block(self.block2, x)
        x = self._forward_block(self.block3, x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

class MaskedReconstructionModel(nn.Module):
    def __init__(self, encoder, in_bands=1290):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(encoder.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, in_bands),
        )

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction

class MaskedSpectralDataset(Dataset):
    def __init__(self, X_np, mask_ratio=0.30):
        self.X = torch.from_numpy(X_np)
        self.mask_ratio = mask_ratio
        self.N, self.bands = self.X.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        target = x.clone()
        num_mask = int(self.bands * self.mask_ratio)
        mask_indices = torch.randperm(self.bands)[:num_mask]
        mask = torch.ones(self.bands, dtype=torch.bool)
        mask[mask_indices] = False
        x[mask_indices] = 0.0
        return x, target, mask

def build_handcrafted_features(X):
    N, B = X.shape
    feats = []
    feats.append(np.mean(X, axis=1, keepdims=True))
    feats.append(np.std(X, axis=1, keepdims=True))
    feats.append(np.median(X, axis=1, keepdims=True))
    feats.append(np.min(X, axis=1, keepdims=True))
    feats.append(np.max(X, axis=1, keepdims=True))
    feats.append(np.percentile(X, 25, axis=1, keepdims=True))
    feats.append(np.percentile(X, 75, axis=1, keepdims=True))
    iqr = np.percentile(X, 75, axis=1, keepdims=True) - np.percentile(X, 25, axis=1, keepdims=True)
    feats.append(iqr)
    band_idx = np.arange(B).reshape(1, -1)
    slope = np.sum((X - X.mean(axis=1, keepdims=True)) * (band_idx - band_idx.mean()), axis=1, keepdims=True)
    slope /= (np.sum((band_idx - band_idx.mean()) ** 2) + 1e-8)
    feats.append(slope)
    deriv1 = np.diff(X, axis=1)
    feats.append(np.mean(deriv1, axis=1, keepdims=True))
    feats.append(np.std(deriv1, axis=1, keepdims=True))
    feats.append(np.max(np.abs(deriv1), axis=1, keepdims=True))
    deriv2 = np.diff(deriv1, axis=1)
    feats.append(np.mean(deriv2, axis=1, keepdims=True))
    feats.append(np.std(deriv2, axis=1, keepdims=True))
    feats.append(np.max(np.abs(deriv2), axis=1, keepdims=True))
    chunk_size = B // 10
    for i in range(10):
        start = i * chunk_size
        end = start + chunk_size if i < 9 else B
        feats.append(np.mean(X[:, start:end], axis=1, keepdims=True))
    from scipy.stats import skew, kurtosis
    feats.append(skew(X, axis=1).reshape(-1, 1))
    feats.append(kurtosis(X, axis=1).reshape(-1, 1))
    feats.append(np.sum(X ** 2, axis=1, keepdims=True))
    feats.append(np.sum(np.abs(deriv1), axis=1, keepdims=True))
    handcrafted = np.concatenate(feats, axis=1).astype(np.float32)
    mu = np.mean(handcrafted, axis=0, keepdims=True)
    sigma = np.std(handcrafted, axis=0, keepdims=True) + 1e-8
    handcrafted = (handcrafted - mu) / sigma
    return handcrafted

def extract_ssl_features(encoder, X_np, device, batch_size=128):
    encoder.eval()
    encoder.to(device)
    X_t = torch.from_numpy(X_np)
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch = X_t[i:i+batch_size].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                feats = encoder(batch)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)

def run_export():
    device = get_device()
    print(f"Exporting Phase 2 features on {device}...")
    data_dir = Path('data')
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    X_raw = hsi_data['X']
    y = hsi_data['y']
    X_hsi = np.nanmean(X_raw, axis=(2, 3)).astype(np.float32)
    del X_raw
    gc.collect()
    col_median = np.nanmedian(X_hsi, axis=0)
    nan_idx = np.where(np.isnan(X_hsi))
    if len(nan_idx[0]) > 0:
        X_hsi[nan_idx] = np.take(col_median, nan_idx[1])
    np.clip(X_hsi, -5.0, 5.0, out=X_hsi)
    msi_data = np.load(data_dir / 'train_msi.npz')
    X_msi = np.nanmean(msi_data['X'], axis=(2, 3)).astype(np.float32)
    msi_mu = np.mean(X_msi, axis=0, keepdims=True)
    msi_std = np.std(X_msi, axis=0, keepdims=True) + 1e-8
    X_msi = ((X_msi - msi_mu) / msi_std).astype(np.float32)
    X_hc = build_handcrafted_features(X_hsi)
    ssl_dataset = MaskedSpectralDataset(X_hsi, mask_ratio=0.30)
    ssl_loader = DataLoader(ssl_dataset, batch_size=128, shuffle=True, drop_last=True)
    encoder = SpectralEncoder1D(in_bands=X_hsi.shape[1], embed_dim=256)
    ssl_model = MaskedReconstructionModel(encoder, in_bands=X_hsi.shape[1]).to(device)
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    print("Training SSL for 100 epochs...")
    ssl_model.train()
    best_loss = float('inf')
    best_state = None
    for epoch in range(100):
        epoch_loss = 0
        for b_x, b_target, b_mask in ssl_loader:
            b_x, b_target, b_mask = b_x.to(device), b_target.to(device), b_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                recon = ssl_model(b_x)
                loss = F.mse_loss(recon[~b_mask], b_target[~b_mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(ssl_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        if (epoch + 1) % 20 == 0: print(f"Epoch {epoch+1}/100, Loss: {avg_loss:.6f}")
    encoder.load_state_dict(best_state)
    torch.save(best_state, data_dir / 'encoder_phase2.pth')
    ssl_features = extract_ssl_features(encoder, X_hsi, device)
    X_final = np.concatenate([ssl_features, X_hc, X_msi], axis=1).astype(np.float32)
    print(f"Saving Cloud Features: {X_final.shape}")
    np.savez_compressed(data_dir / 'train_fused_phase2.npz', X=X_final, y=y)
    print("Export Complete.")

if __name__ == '__main__':
    run_export()
