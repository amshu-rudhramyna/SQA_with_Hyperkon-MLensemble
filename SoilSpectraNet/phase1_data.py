import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import os
from scipy.signal import savgol_filter

def apply_snv(spectra):
    mean = np.nanmean(spectra, axis=0, keepdims=True)
    std = np.nanstd(spectra, axis=0, keepdims=True) + 1e-8
    return (spectra - mean) / std

def get_mean_spectrum(arrays):
    sum_spec = 0
    count = 0
    for arr in arrays:
        # Ignore NaNs
        sum_spec += np.nansum(arr, axis=(1, 2))
        valid_count = np.sum(~np.isnan(arr[0])) # Non-nan pixels
        count += valid_count
    return sum_spec / (count + 1e-8)

def apply_msc(spectra, ref_spectrum):
    B, H, W = spectra.shape
    reshaped = spectra.reshape(B, -1)
    
    # Only fit on valid pixels
    valid_mask = ~np.isnan(reshaped[0])
    msc_spectra = np.copy(reshaped)
    
    # We iter over valid pixels. Usually, can vectorize. 
    # For simplicity and speed:
    ref_mean = np.mean(ref_spectrum)
    ref_var = np.var(ref_spectrum)
    
    spec_mean = np.mean(reshaped[:, valid_mask], axis=0) # (valid_pixels,)
    cov = np.mean((reshaped[:, valid_mask] - spec_mean) * (ref_spectrum[:, None] - ref_mean), axis=0)
    
    b = cov / (ref_var + 1e-8)
    a = spec_mean - b * ref_mean
    
    msc_spectra[:, valid_mask] = (reshaped[:, valid_mask] - a) / (b + 1e-8)
    return msc_spectra.reshape(B, H, W)

def pad_array(arr, max_h, max_w):
    h, w = arr.shape[1], arr.shape[2]
    pad_h = max(0, max_h - h)
    pad_w = max(0, max_w - w)
    # Pad with NaNs
    return np.pad(arr, ((0,0), (pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant', constant_values=np.nan)

def process_file_phase1(idx_row, base_dir, targets, max_h, max_w):
    i, row = idx_row
    idx = int(row['sample_index'])
    fname = f"{idx:04d}.npz"
    fpath = os.path.join(base_dir, fname)
    
    if not os.path.exists(fpath):
        return None
        
    data = np.load(fpath)
    arr = data['data'] if 'data' in data else data['arr_0']
    arr = arr.astype(np.float32)
    B, H, W = arr.shape
    
    # 1. Spectral Unmixing: NDVI Threshold masking
    # HSI Airborne: Red is ~Band 77, NIR is ~Band 134
    red = arr[77, :, :]
    nir = arr[134, :, :]
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    # "Remove non-soil pixels using soil mask"
    soil_mask = ndvi < 0.25 # Typical threshold. >0.25 is live vegetation
    
    # If a field is entirely vegetation (unlikely, but possible), keep top 5% lowest NDVI pixels
    if not np.any(soil_mask):
         threshold = np.percentile(ndvi, 5)
         soil_mask = ndvi <= threshold

    # Apply mask (set non-soil to NaN)
    arr[:, ~soil_mask] = np.nan
    
    # Flatten purely for spectral derivations
    reshaped = arr.reshape(B, -1)
    
    # 2. Savitzky-Golay Filtering & Derivatives
    # Window length 11, polynomial order 2
    # We apply SG filter along spectral axis (axis=0)
    # Savgol cannot handle NaNs directly, so we run it only on valid pixels
    valid_pixels = reshaped[:, soil_mask.flatten()]
    
    if valid_pixels.shape[1] > 0:
        smoothed = savgol_filter(valid_pixels, window_length=15, polyorder=2, axis=0)
        deriv1 = savgol_filter(valid_pixels, window_length=15, polyorder=2, deriv=1, axis=0)
        deriv2 = savgol_filter(valid_pixels, window_length=15, polyorder=2, deriv=2, axis=0)
        
        # Concatenate features
        combined_features = np.concatenate([smoothed, deriv1, deriv2], axis=0)
    else:
        # Fallback if somehow completely broken
        combined_features = np.full((B*3, valid_pixels.shape[1]), np.nan, dtype=np.float32)
    
    new_B = combined_features.shape[0]
    out_arr = np.full((new_B, H*W), np.nan, dtype=np.float32)
    out_arr[:, soil_mask.flatten()] = combined_features
    
    out_arr = out_arr.reshape(new_B, H, W)
    out_arr = pad_array(out_arr, max_h, max_w)
    
    y = row[targets].values.astype(np.float32)
    return idx, out_arr, y

def build_phase1_data():
    print("Building Phase 1 Features: HSI Airborne with Savgol, Derivs, and NDVI Masking")
    train_dir = Path('c:/Users/One Page VR/Desktop/hypersoilnet2/data/raw/HYPERVIEW2/train')
    csv_path = train_dir.parent / 'train_gt.csv'
    
    df = pd.read_csv(csv_path)
    targets = ['P', 'K', 'Mg', 'pH']
    if not all(t in df.columns for t in targets):
        targets = ['B', 'Cu', 'Zn', 'Fe']
    
    base_dir = train_dir / 'hsi_airborne'
    max_h, max_w = 20, 20
    
    with Pool(cpu_count()) as pool:
        results = pool.map(
            partial(process_file_phase1, base_dir=base_dir, targets=targets, max_h=max_h, max_w=max_w), 
            list(df.iterrows())
        )
    
    results = [r for r in results if r is not None]
    idxs = np.array([r[0] for r in results])
    arrays = [r[1] for r in results]
    ys = np.stack([r[2] for r in results])
    
    # Extract SNV + MSC
    print("Applying SNV + MSC normalization arrays...")
    final_arrays = []
    
    # Global MSC requires processing chunk by chunk to avoid RAM limits with 1290 features, but let's try
    stacked = np.stack(arrays) 
    
    # SNV
    stacked = apply_snv(stacked)
    
    # MSC
    ref_spec = get_mean_spectrum(stacked)
    
    # Apply MSC in parallel chunks to save time
    with Pool(cpu_count()) as pool:
        msc_arrays = pool.starmap(apply_msc, [(arr, ref_spec) for arr in stacked])
        
    final_arr = np.stack(msc_arrays)
    
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / 'train_hsi_phase1.npz', X=final_arr, y=ys, idx=idxs)
    print(f"Phase 1 Dataset Saved. Shape: {final_arr.shape}")

if __name__ == '__main__':
    build_phase1_data()
