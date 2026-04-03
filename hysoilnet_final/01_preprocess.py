import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pywt
import json, glob, os

TARGETS   = ['B', 'Fe', 'Zn', 'Cu', 'Mn', 'S']
LOG_TARGETS = ['B', 'Cu', 'Zn', 'Mn', 'S']
N_SOIL_CLUSTERS = 5

def load_wavelengths(wl_json_path):
    with open(wl_json_path) as f:
        wl = json.load(f)
    return wl

def extract_masked_mean_spectrum(data, mask):
    # data: (bands, H, W)  mask: (bands, H, W) or (H, W)
    if mask.ndim == 2:
        return np.array([data[b][mask].mean() for b in range(data.shape[0])])
    else:
        return np.array([
            data[b][mask[b]].mean() if mask[b].any() else np.nan
            for b in range(data.shape[0])
        ])

def resample_to_150(spectrum, wl_source, wl_target_150):
    valid = ~np.isnan(spectrum)
    if valid.sum() < 10:
        return np.full(150, np.nan, dtype=np.float32)
    interp = interp1d(wl_source[valid], spectrum[valid],
                      kind='linear', bounds_error=False, fill_value='extrapolate')
    return interp(wl_target_150).astype(np.float32)

def load_airborne(data_dir, wl_airborne, wl_target_150, targets):
    spectra, labels_list = [], []
    
    gt_df = pd.read_csv(os.path.join(data_dir, "train_gt.csv"))
    gt_df.set_index('sample_index', inplace=True)
    
    for f in sorted(glob.glob(f"{data_dir}/train/hsi_airborne/*.npz")):
        idx = int(os.path.basename(f).split('.')[0])
        if idx not in gt_df.index:
            continue
            
        d    = np.load(f, allow_pickle=True)
        data = d['data'].astype(np.float32) / 10000.0  # DN → reflectance
        mask = d['mask']
        spec_native = extract_masked_mean_spectrum(data, mask)  # (430,)
        spec_150    = resample_to_150(spec_native, wl_airborne, wl_target_150)  # (150,)
        
        if np.isnan(spec_150).all():
            continue
            
        spectra.append(spec_150)
        row = gt_df.loc[idx]
        labels_list.append({t: float(row[t]) for t in targets})
        
    X = np.array(spectra, dtype=np.float32)
    Y = {t: np.array([l[t] for l in labels_list], dtype=np.float32) for t in targets}
    return X, Y

def log_transform(Y):
    Y_log = {}
    for t in TARGETS:
        if t in LOG_TARGETS:
            Y_log[t] = np.log1p(Y[t])
        else:
            Y_log[t] = Y[t].copy()
    return Y_log

def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

def continuum_removal(spectrum):
    wl = np.arange(len(spectrum))
    pts = np.column_stack([wl, spectrum])
    try:
        hull = ConvexHull(pts)
        hull_idx = np.sort(hull.vertices)
        hull_refl = np.interp(wl, wl[hull_idx], spectrum[hull_idx])
        return spectrum / (hull_refl + 1e-8)
    except Exception:
        return spectrum

def savitzky_golay_derivatives(spectrum, window=11, poly=3):
    from scipy.signal import savgol_filter
    d1 = savgol_filter(spectrum, window, poly, deriv=1)
    d2 = savgol_filter(spectrum, window, poly, deriv=2)
    return d1, d2

def dwt_features(spectrum, wavelet='dmey', level=4):
    coeffs = pywt.wavedec(spectrum, wavelet, level=level)
    return np.concatenate([c for c in coeffs])

def svd_features(X, n_components=20):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components] * s[:n_components]

def fft_features(spectrum):
    f = np.abs(np.fft.rfft(spectrum))
    return f[:len(f)//2]

def geochemical_band_weights(n_bands=150):
    weights = np.ones(n_bands, dtype=np.float32)
    weights[0:98]   *= 1.5   # SOM proxy (462–560nm)
    # The slicing index [400:440] was larger than 150 which will not apply weights. Leaving as original script. 
    return weights

def build_feature_matrix(X):
    X_snv = snv(X)
    features = [X_snv]
    cr_feats, d1_feats, d2_feats, dwt_feats, fft_feats = [], [], [], [], []
    for s in X_snv:
        cr  = continuum_removal(s)
        d1, d2 = savitzky_golay_derivatives(s)
        dw  = dwt_features(s)
        ff  = fft_features(s)
        cr_feats.append(cr); d1_feats.append(d1)
        d2_feats.append(d2); dwt_feats.append(dw); fft_feats.append(ff)
    features += [
        np.array(cr_feats),
        np.array(d1_feats),
        np.array(d2_feats),
        np.array(dwt_feats),
        np.array(fft_feats),
    ]
    svd_f = svd_features(X_snv)
    features.append(svd_f)
    return np.concatenate(features, axis=1).astype(np.float32)

def get_soil_clusters(X):
    km = KMeans(n_clusters=N_SOIL_CLUSTERS, random_state=42, n_init=10)
    return km.fit_predict(X)

if __name__ == '__main__':
    import pickle
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'raw', 'HYPERVIEW2')
    wl_json_path = os.path.join(data_dir, 'wavelengths.json')
    
    # Load exact 430 bands from airborne dataset
    wl_dict = load_wavelengths(wl_json_path)
    wl_airborne = np.array(list(wl_dict['hsi_aerial_wavelengths'].values()))
    
    # Target 150-band wavelength grid
    WL_TARGET_150 = np.linspace(462, 942, 150)
    
    print("Loading exactly from airborne dataset...")
    X, Y = load_airborne(data_dir, wl_airborne, WL_TARGET_150, TARGETS)
    
    Y_log    = log_transform(Y)
    X_feat   = build_feature_matrix(X)
    clusters = get_soil_clusters(X)
    bw       = geochemical_band_weights()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    
    os.makedirs('cache/', exist_ok=True)
    np.save('cache/X_raw.npy',    X)
    np.save('cache/X_feat.npy',   X_scaled)
    np.save('cache/clusters.npy', clusters)
    np.save('cache/band_weights.npy', bw)
    with open('cache/Y.pkl',     'wb') as f: pickle.dump(Y,     f)
    with open('cache/Y_log.pkl', 'wb') as f: pickle.dump(Y_log, f)
    with open('cache/scaler.pkl','wb') as f: pickle.dump(scaler, f)
    print(f"Preprocessed {len(X)} samples | Feature dim: {X_scaled.shape[1]}")

