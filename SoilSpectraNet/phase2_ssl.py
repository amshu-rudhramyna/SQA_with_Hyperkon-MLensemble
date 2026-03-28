import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# DATASETS
# ===================================================================

class MaskedSpectralDataset(Dataset):
    """
    Masked Band Reconstruction pretext task.
    Randomly masks 30% of spectral bands and trains model to reconstruct them.
    """
    def __init__(self, X_np, mask_ratio=0.30):
        self.X = torch.from_numpy(X_np)  # Zero-copy
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


# ===================================================================
# 1D CNN ENCODER (same architecture as before)
# ===================================================================

class SpectralEncoder1D(nn.Module):
    """Deep 1D CNN backbone, outputs 256-dim features."""
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
    """Encoder + lightweight decoder for masked band reconstruction."""
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


# ===================================================================
# HANDCRAFTED FEATURES
# ===================================================================

def build_handcrafted_features(X):
    """Build statistical + derivative features from spectral vectors."""
    N, B = X.shape
    feats = []

    # Global statistics (5)
    feats.append(np.mean(X, axis=1, keepdims=True))
    feats.append(np.std(X, axis=1, keepdims=True))
    feats.append(np.median(X, axis=1, keepdims=True))
    feats.append(np.min(X, axis=1, keepdims=True))
    feats.append(np.max(X, axis=1, keepdims=True))

    # Quartiles (3)
    feats.append(np.percentile(X, 25, axis=1, keepdims=True))
    feats.append(np.percentile(X, 75, axis=1, keepdims=True))
    iqr = np.percentile(X, 75, axis=1, keepdims=True) - np.percentile(X, 25, axis=1, keepdims=True)
    feats.append(iqr)

    # Spectral slope (1)
    band_idx = np.arange(B).reshape(1, -1)
    slope = np.sum((X - X.mean(axis=1, keepdims=True)) * (band_idx - band_idx.mean()), axis=1, keepdims=True)
    slope /= (np.sum((band_idx - band_idx.mean()) ** 2) + 1e-8)
    feats.append(slope)

    # First derivative stats (3)
    deriv1 = np.diff(X, axis=1)
    feats.append(np.mean(deriv1, axis=1, keepdims=True))
    feats.append(np.std(deriv1, axis=1, keepdims=True))
    feats.append(np.max(np.abs(deriv1), axis=1, keepdims=True))

    # Second derivative stats (3)
    deriv2 = np.diff(deriv1, axis=1)
    feats.append(np.mean(deriv2, axis=1, keepdims=True))
    feats.append(np.std(deriv2, axis=1, keepdims=True))
    feats.append(np.max(np.abs(deriv2), axis=1, keepdims=True))

    # Band-chunk means (10 segments) (10)
    chunk_size = B // 10
    for i in range(10):
        start = i * chunk_size
        end = start + chunk_size if i < 9 else B
        feats.append(np.mean(X[:, start:end], axis=1, keepdims=True))

    # Skewness and kurtosis (2)
    from scipy.stats import skew, kurtosis
    feats.append(skew(X, axis=1).reshape(-1, 1))
    feats.append(kurtosis(X, axis=1).reshape(-1, 1))

    # Energy / power features (2)
    feats.append(np.sum(X ** 2, axis=1, keepdims=True))  # Total energy
    feats.append(np.sum(np.abs(deriv1), axis=1, keepdims=True))  # Total variation

    handcrafted = np.concatenate(feats, axis=1).astype(np.float32)

    # Normalize
    mu = np.mean(handcrafted, axis=0, keepdims=True)
    sigma = np.std(handcrafted, axis=0, keepdims=True) + 1e-8
    handcrafted = (handcrafted - mu) / sigma

    return handcrafted


# ===================================================================
# FEATURE EXTRACTION
# ===================================================================

def extract_ssl_features(encoder, X_np, device, batch_size=128):
    """Extract SSL features from the pretrained encoder for all samples."""
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
            del batch, feats
    
    return np.concatenate(all_feats, axis=0)


# ===================================================================
# TARGET-WISE XGBOOST ENSEMBLE
# ===================================================================

def get_target_ensemble(target_idx):
    """
    Target-wise stacking ensemble with tuned hyperparameters per target.
    Each target gets its own optimized ensemble.
    """
    # Base models with reasonable defaults for N~1700 samples
    base_models = [
        ('rf', RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            max_features='sqrt', n_jobs=-1, random_state=42
        )),
        ('xgb', XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42
        )),
    ]
    
    meta_learner = Ridge(alpha=1.0)
    return StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,  # Internal CV for stacking
        n_jobs=1,
        passthrough=True,  # Also pass original features to meta-learner
    )


# ===================================================================
# MAIN
# ===================================================================

def run_phase_2():
    device = get_device()
    print(f"[Phase 2 SSL v3 — High Impact] Device: {device}")

    data_dir = Path('data')
    
    # -------------------------------------------------------------------
    # LOAD & POOL HSI DATA
    # -------------------------------------------------------------------
    print("Loading HSI data...")
    hsi_data = np.load(data_dir / 'train_hsi_phase1.npz')
    X_raw = hsi_data['X']  # (1876, 1290, 20, 20)
    y = hsi_data['y']      # (1876, 4)

    print("Spatial NaN-aware mean pooling HSI...")
    X_hsi = np.nanmean(X_raw, axis=(2, 3)).astype(np.float32)  # (1876, 1290)
    del X_raw
    gc.collect()

    # Clean
    col_median = np.nanmedian(X_hsi, axis=0)
    nan_idx = np.where(np.isnan(X_hsi))
    if len(nan_idx[0]) > 0:
        X_hsi[nan_idx] = np.take(col_median, nan_idx[1])
    np.clip(X_hsi, -5.0, 5.0, out=X_hsi)
    X_hsi = np.ascontiguousarray(X_hsi, dtype=np.float32)

    # -------------------------------------------------------------------
    # LOAD & POOL MSI DATA (MULTI-MODAL FUSION)
    # -------------------------------------------------------------------
    print("Loading MSI satellite data for multi-modal fusion...")
    msi_data = np.load(data_dir / 'train_msi.npz')
    X_msi_raw = msi_data['X']  # (1876, 12, 8, 8)
    X_msi = np.nanmean(X_msi_raw, axis=(2, 3)).astype(np.float32)  # (1876, 12)
    del X_msi_raw, msi_data
    gc.collect()

    # Normalize MSI
    msi_mu = np.mean(X_msi, axis=0, keepdims=True)
    msi_std = np.std(X_msi, axis=0, keepdims=True) + 1e-8
    X_msi = ((X_msi - msi_mu) / msi_std).astype(np.float32)

    print(f"HSI spectral: {X_hsi.shape} ({X_hsi.nbytes / 1e6:.1f} MB)")
    print(f"MSI spectral: {X_msi.shape} ({X_msi.nbytes / 1e3:.1f} KB)")
    print(f"Targets: {y.shape}")

    # -------------------------------------------------------------------
    # BUILD HANDCRAFTED FEATURES
    # -------------------------------------------------------------------
    print("Building handcrafted spectral features (HSI)...")
    X_hc = build_handcrafted_features(X_hsi)
    print(f"Handcrafted features: {X_hc.shape}")

    # -------------------------------------------------------------------
    # PART 1: MASKED BAND RECONSTRUCTION SSL PRETRAINING
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[PART 1/3] Self-Supervised Masked Band Reconstruction")
    print("=" * 60)

    ssl_dataset = MaskedSpectralDataset(X_hsi, mask_ratio=0.30)
    ssl_loader = DataLoader(ssl_dataset, batch_size=128, shuffle=True,
                            pin_memory=False, num_workers=0, drop_last=True)

    encoder = SpectralEncoder1D(in_bands=X_hsi.shape[1], embed_dim=256)
    ssl_model = MaskedReconstructionModel(encoder, in_bands=X_hsi.shape[1]).to(device)

    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    ssl_epochs = 100
    ssl_model.train()
    best_ssl_loss = float('inf')
    best_encoder_state = None
    start_time = time.time()

    for epoch in range(ssl_epochs):
        epoch_loss = 0
        n_batches = 0
        for b_x, b_target, b_mask in ssl_loader:
            b_x = b_x.to(device, non_blocking=True)
            b_target = b_target.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    recon = ssl_model(b_x)
                    masked_loss = F.mse_loss(recon[~b_mask], b_target[~b_mask])
                scaler.scale(masked_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = ssl_model(b_x)
                masked_loss = F.mse_loss(recon[~b_mask], b_target[~b_mask])
                masked_loss.backward()
                optimizer.step()

            epoch_loss += masked_loss.item()
            n_batches += 1
            del b_x, b_target, b_mask, recon, masked_loss

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_ssl_loss:
            best_ssl_loss = avg_loss
            best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"SSL Epoch {epoch+1}/{ssl_epochs} | Recon Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"SSL Pretraining complete in {time.time() - start_time:.1f}s (best loss: {best_ssl_loss:.6f})")

    # Save best encoder
    torch.save(best_encoder_state, data_dir / 'encoder_phase2.pth')
    
    # Load best weights
    encoder.load_state_dict(best_encoder_state)

    # Cleanup
    del ssl_model, ssl_dataset, ssl_loader, optimizer, scheduler, scaler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # PART 2: EXTRACT SSL FEATURES
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[PART 2/3] Extracting SSL Features for Ensemble")
    print("=" * 60)

    ssl_features = extract_ssl_features(encoder, X_hsi, device)
    print(f"SSL features: {ssl_features.shape}")

    # Free GPU encoder
    encoder.cpu()
    del encoder
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Build final feature matrix: SSL (256) + Handcrafted (29) + MSI (12) = ~297 features
    X_final = np.concatenate([ssl_features, X_hc, X_msi], axis=1).astype(np.float32)
    print(f"Final fused feature matrix: {X_final.shape} "
          f"(SSL:{ssl_features.shape[1]} + HC:{X_hc.shape[1]} + MSI:{X_msi.shape[1]})")

    del ssl_features, X_hc, X_msi  # Free intermediate arrays
    gc.collect()

    # Save features for Phase 3
    print(f"Saving fused features for Phase 3 to {data_dir / 'train_fused_phase2.npz'}...")
    np.savez_compressed(data_dir / 'train_fused_phase2.npz', X=X_final, y=y)

    # -------------------------------------------------------------------
    # PART 3: TARGET-WISE XGBOOST STACKING ENSEMBLE (10-FOLD CV)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[PART 3/3] Target-Wise Stacking Ensemble — 10-Fold CV")
    print("=" * 60)

    y = np.ascontiguousarray(y, dtype=np.float32)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    targets_count = y.shape[1]
    target_names = ['P', 'K', 'Mg', 'pH']
    all_r2 = {t: [] for t in range(targets_count)}

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_final)):
        X_train, X_val = X_final[train_idx], X_final[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_r2s = []
        for t in range(targets_count):
            model = get_target_ensemble(t)
            model.fit(X_train, y_train[:, t])
            preds = model.predict(X_val)

            r2 = r2_score(y_val[:, t], preds)
            all_r2[t].append(r2)
            fold_r2s.append(r2)

        print(f"Fold {fold+1}/10 | " + " | ".join(
            f"{target_names[t]}={fold_r2s[t]:.4f}" for t in range(targets_count)
        ) + f" | Avg={np.mean(fold_r2s):.4f}")

    total_time = time.time() - start_time

    # Final Report
    print("\n" + "=" * 60)
    print("PHASE 2 LOG: SSL + XGBOOST ENSEMBLE + MULTI-MODAL (10-Fold CV)")
    print("=" * 60)
    avg_r2_across_targets = []
    for t in range(targets_count):
        mean_r2 = np.mean(all_r2[t])
        std_r2 = np.std(all_r2[t])
        avg_r2_across_targets.append(mean_r2)
        print(f"{target_names[t]:>4s}: R² = {mean_r2:.4f} ± {std_r2:.4f}")

    print(f"\nGLOBAL PHASE 2 AVERAGE R²: {np.mean(avg_r2_across_targets):.4f}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == '__main__':
    run_phase_2()
