import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_data():
    base_dir = Path('../data/raw/HYPERVIEW2/train/hsi_airborne')
    csv_path = Path('../data/raw/HYPERVIEW2/train_gt.csv')
    
    if not csv_path.exists():
        print(f"Error: Could not find ground truth at {csv_path.resolve()}")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    targets = ['B', 'Cu', 'Zn', 'Fe']
    
    X_list = []
    y_list = []
    
    print(f"Loading raw spectral data from {base_dir}...")
    valid_idx = []
    for i, row in df.iterrows():
        fname = str(int(row['sample_index'])).zfill(4) + '.npz'
        fpath = base_dir / fname
        if fpath.exists():
            data = np.load(fpath)
            arr = data['data'] if 'data' in data else data['arr_0']
            
            if arr.ndim == 3:
                spectrum = np.mean(arr, axis=(1, 2))
            elif arr.ndim == 2:
                spectrum = np.mean(arr, axis=(0,)) if arr.shape[0] < arr.shape[1] else np.mean(arr, axis=(1,))
            else:
                spectrum = arr.copy()
            
            if spectrum.shape[0] > 150:
                spectrum = spectrum[:150]
            elif spectrum.shape[0] < 150:
                spectrum = np.pad(spectrum, (0, 150 - spectrum.shape[0]))
                
            X_list.append(spectrum)
            y_list.append(row[targets].values.astype(np.float32))
            valid_idx.append(i)
        
        if len(valid_idx) % 100 == 0 and len(valid_idx) > 0 and (len(valid_idx) == 100 or i == len(df)-1):
            print(f"Loaded {len(valid_idx)} samples...")

    X = np.stack(X_list)
    y = np.stack(y_list)
    
    print(f"Total parsed shape: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    out_dir = Path('data')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(out_dir / 'train.npz', X=X_train, y=y_train)
    np.savez(out_dir / 'test.npz', X=X_test, y=y_test)
    
    print(f"Saved parsed data to {out_dir.resolve()}")

if __name__ == '__main__':
    prepare_data()
