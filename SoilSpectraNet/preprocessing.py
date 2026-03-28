import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import os

def apply_snv(spectra):
    """
    Apply Standard Normal Variate (SNV) to a 3D array (Bands, H, W).
    Normalizes each pixel's spectrum to mean 0 and std 1.
    """
    mean = np.mean(spectra, axis=0, keepdims=True)
    std = np.std(spectra, axis=0, keepdims=True) + 1e-8
    return (spectra - mean) / std

def get_mean_spectrum(arrays):
    """
    Computes global mean spectrum across a list of (B, H, W) arrays.
    """
    sum_spec = 0
    count = 0
    for arr in arrays:
        # Sum over spatial dims
        sum_spec += np.sum(arr, axis=(1, 2))
        count += arr.shape[1] * arr.shape[2]
    return sum_spec / count

def apply_msc(spectra, ref_spectrum):
    """
    Apply Multiplicative Scatter Correction (MSC) to a 3D array (B, H, W).
    spectra: (B, H, W)
    ref_spectrum: (B,)
    """
    B, H, W = spectra.shape
    reshaped = spectra.reshape(B, -1) # (B, H*W)
    
    # Fit line: spectra = a + b * ref
    # To do this efficiently:
    # slope b = cov(spectra, ref) / var(ref)
    ref_mean = np.mean(ref_spectrum)
    ref_var = np.var(ref_spectrum)
    
    spec_mean = np.mean(reshaped, axis=0) # (H*W,)
    
    cov = np.mean((reshaped - spec_mean) * (ref_spectrum[:, None] - ref_mean), axis=0) # (H*W,)
    b = cov / (ref_var + 1e-8)
    a = spec_mean - b * ref_mean
    
    msc_spectra = (reshaped - a) / (b + 1e-8)
    return msc_spectra.reshape(B, H, W)

def pad_array(arr, max_h, max_w):
    h, w = arr.shape[1], arr.shape[2]
    pad_h = max(0, max_h - h)
    pad_w = max(0, max_w - w)
    
    # Pad symmetrically if possible, else 0 pad
    return np.pad(arr, ((0,0), (pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')

def process_file(idx_row, base_dir, targets, max_h, max_w, apply_snv_flag=True):
    i, row = idx_row
    idx = int(row['sample_index'])
    fname = f"{idx:04d}.npz"
    fpath = os.path.join(base_dir, fname)
    
    if not os.path.exists(fpath):
        return None
        
    data = np.load(fpath)
    arr = data['data'] if 'data' in data else data['arr_0']
    
    # Cast float32
    arr = arr.astype(np.float32)
    
    if apply_snv_flag:
        arr = apply_snv(arr)
        
    arr = pad_array(arr, max_h, max_w)
    
    y = row[targets].values.astype(np.float32)
    return idx, arr, y

def process_modality(df, base_dir, max_h, max_w, targets, name):
    print(f"Processing {name} with multiprocessing ({cpu_count()} cores)...")
    
    with Pool(cpu_count()) as pool:
        results = pool.map(
            partial(process_file, base_dir=base_dir, targets=targets, max_h=max_h, max_w=max_w), 
            list(df.iterrows())
        )
    
    # Filter missing
    results = [r for r in results if r is not None]
    
    idxs = np.array([r[0] for r in results])
    arrays = [r[1] for r in results]
    ys = np.stack([r[2] for r in results])
    
    # MSC
    print(f"Applying MSC for {name}...")
    ref_spectrum = get_mean_spectrum(arrays)
    
    with Pool(cpu_count()) as pool:
        msc_arrays = pool.starmap(apply_msc, [(arr, ref_spectrum) for arr in arrays])
        
    final_arr = np.stack(msc_arrays)
    return idxs, final_arr, ys

if __name__ == '__main__':
    train_dir = Path('c:/Users/One Page VR/Desktop/hypersoilnet2/data/raw/HYPERVIEW2/train')
    csv_path = train_dir.parent / 'train_gt.csv'
    
    df = pd.read_csv(csv_path)
    targets = ['P', 'K', 'Mg', 'pH']
    # Let me check the real targets from df
    if not all(t in df.columns for t in targets):
        # Fallback to the other known targets
        targets = ['B', 'Cu', 'Zn', 'Fe'] # used in R2 0.85
        if not all(t in df.columns for t in targets):
             targets = [c for c in df.columns if c not in ['sample_index']]
    print(f"Targets: {targets}")
    
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)
    
    # Process Airborne (Max pad ~ 20x20)
    idxs_hsi, arr_hsi, y_hsi = process_modality(df, train_dir / 'hsi_airborne', 20, 20, targets, 'HSI_Airborne')
    np.savez_compressed(out_dir / 'train_hsi.npz', X=arr_hsi, y=y_hsi, idx=idxs_hsi)
    
    # Process MSI (Max pad ~ 8x8)
    idxs_msi, arr_msi, y_msi = process_modality(df, train_dir / 'msi_satellite', 8, 8, targets, 'MSI_Satellite')
    np.savez_compressed(out_dir / 'train_msi.npz', X=arr_msi, y=y_msi, idx=idxs_msi)
    
    # Process PRISMA (Max pad ~ 4x4)
    idxs_prisma, arr_prisma, y_prisma = process_modality(df, train_dir / 'hsi_satellite', 4, 4, targets, 'PRISMA')
    np.savez_compressed(out_dir / 'train_prisma.npz', X=arr_prisma, y=y_prisma, idx=idxs_prisma)
    
    print("Saving completed.")
