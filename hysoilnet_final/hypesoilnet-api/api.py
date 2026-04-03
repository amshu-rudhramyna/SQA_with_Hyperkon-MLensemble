import os
import json
import io
import pickle
import numpy as np
import pywt
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from scipy.interpolate import interp1d
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity

# --- CONSTANTS & CONFIG ---
TARGETS = ['B', 'Fe', 'Zn', 'Cu', 'Mn', 'S']
LOG_TARGETS = ['B', 'Cu', 'Zn', 'Mn', 'S']
UNITS = {'B': 'mg/kg', 'Fe': 'g/kg', 'Zn': 'mg/kg', 'Cu': 'mg/kg', 'Mn': 'mg/kg', 'S': 'g/kg'}

app = FastAPI(title="HyperSoilNet API")

# Enable CORS for the static GitHub Pages UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL DEFINITIONS ---
MTL_WEIGHTS = {'B':1.0,'Fe':1.0,'Zn':1.0,'Cu':1.5,'Mn':1.0,'S':1.0,'SOM':0.2}

class CBAMBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.ch_avg = nn.AdaptiveAvgPool1d(1)
        self.ch_max = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//r, bias=False), nn.ReLU(),
            nn.Linear(ch//r, ch, bias=False))
        self.sp_conv = nn.Conv1d(2, 1, 7, padding=3, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = self.fc(self.ch_avg(x).squeeze(-1))
        b = self.fc(self.ch_max(x).squeeze(-1))
        x = x * self.sig(a + b).unsqueeze(-1)
        sp = torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True).values], 1)
        return x * self.sig(self.sp_conv(sp))

class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch))
        self.attn = CBAMBlock(ch)
        self.act  = nn.ReLU()

    def forward(self, x):
        return self.act(self.net(x) + x)

class SpectralCNN(nn.Module):
    def __init__(self, in_bands=150, embed_dim=128, n_targets=6, has_som=True):
        super().__init__()
        self.band_w = nn.Parameter(torch.ones(in_bands), requires_grad=False)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.stage1 = nn.Sequential(ResBlock1D(64), ResBlock1D(64), ResBlock1D(64))
        self.pool1  = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(*[ResBlock1D(128) for _ in range(4)])
        self.pool2  = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        self.stage3 = nn.Sequential(*[ResBlock1D(256) for _ in range(6)])
        self.pool3  = nn.Conv1d(256, embed_dim, 3, stride=2, padding=1)
        self.stage4 = nn.Sequential(*[ResBlock1D(embed_dim) for _ in range(3)])
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.drop   = nn.Dropout(0.3)
        self.heads  = nn.ModuleDict({t: nn.Linear(embed_dim, 1) for t in MTL_WEIGHTS})

    def forward(self, x):
        x = x * self.band_w.unsqueeze(0)
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.stage4(x)
        emb = self.drop(self.gap(x).squeeze(-1))
        # preds = {t: self.heads[t](emb).squeeze(-1) for t in self.heads}
        return emb

# --- GLOBALS & PRELOAD ---
cnn_model = None
ensemble_models = None
correctors = None
scaler = None
svd_Vt = None
wl_native = None
wl_150 = np.linspace(462, 942, 150)

@app.on_event("startup")
def load_models():
    global cnn_model, ensemble_models, correctors, wl_native, scaler, svd_Vt
    print("Loading models into memory...")

    # Load Wavelengths
    try:
        with open("models/wavelengths.json", "r") as f:
            wl_native = np.array(json.load(f)["hsi_airborne"])
    except:
        wl_native = np.linspace(414, 2357, 430)

    # Load CNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = SpectralCNN().to(device)
    cnn_model.load_state_dict(torch.load("models/spectral_cnn.pt", map_location=device))
    cnn_model.eval()

    # Load Ensemble Trees, Correctors & Preprocessors
    with open("models/ensemble.pkl", "rb") as f:
        ensemble_models = pickle.load(f)
    with open("models/correctors.pkl", "rb") as f:
        correctors = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    svd_Vt = np.load("models/svd_Vt.npy")

    print("System Ready.")

# --- PREPROCESSING & FEATURE EXTRACTION ---
from scipy.spatial import ConvexHull

def extract_spectral_features(spectrum):
    # 0. SNV (already performed earlier but we expect SNV scaled spectrum)
    s = spectrum
    
    # 1. Continuum Removal
    try:
        wl = np.arange(len(s))
        pts = np.column_stack([wl, s])
        hull = ConvexHull(pts)
        hull_idx = np.sort(hull.vertices)
        hull_refl = np.interp(wl, wl[hull_idx], s[hull_idx])
        cr = s / (hull_refl + 1e-8)
    except Exception:
        cr = s.copy()
        
    # 2. Savitzky-Golay
    from scipy.signal import savgol_filter
    d1 = savgol_filter(s, 11, 3, deriv=1)
    d2 = savgol_filter(s, 11, 3, deriv=2)
    
    # 3. Wavelets
    coeffs = pywt.wavedec(s, 'dmey', level=4)
    dwt = np.concatenate([c for c in coeffs])
    
    # 4. FFT
    ff = np.abs(np.fft.rfft(s))
    fft = ff[:len(ff)//2]
    
    # 5. SVD Projection
    svd = s.dot(svd_Vt.T)
    
    return np.concatenate([s, cr, d1, d2, dwt, fft, svd])

def process_npz(file_bytes):
    # Load raw .npz
    with io.BytesIO(file_bytes) as f:
        npz = np.load(f)
        keys = list(npz.keys())
        data_key = 'data' if 'data' in keys else keys[0]
        if data_key == 'mask' and len(keys) > 1:
            data_key = keys[1]
            
        cube = npz[data_key]
        
        if 'mask' not in keys:
            mask = np.ones((cube.shape[1], cube.shape[2]), dtype=bool)
        else:
            mask = npz['mask']
            if mask.ndim == 3:
                mask = mask[0]
        
        if cube.shape[0] != 430:
            raise ValueError(f"Expected 430 channels, got {cube.shape[0]}")

        # Mean valid pixel
        valid_pixels = cube[:, mask.astype(bool)]
        if valid_pixels.shape[1] == 0:
             mean_sig = np.mean(cube, axis=(1,2)) 
        else:
             mean_sig = np.mean(valid_pixels, axis=1)

        # Scale reflectance -> interp to 150
        y_sig = mean_sig / 10000.0
        f_int = interp1d(wl_native, y_sig, kind='linear', bounds_error=False, fill_value='extrapolate')
        return f_int(wl_150)

# --- INFERENCE PIPELINE ---
def predict_soil(spectrum_150):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. SNV & CNN Features
    mu, std = spectrum_150.mean(), spectrum_150.std() + 1e-12
    snv = (spectrum_150 - mu) / std
    
    with torch.no_grad():
        x_in = torch.tensor(snv, dtype=torch.float32).unsqueeze(0).to(device)
        cnn_feat = cnn_model(x_in).cpu().numpy().flatten()
    
    # 2. Handcrafted Features & Scaling
    hand_feat = extract_spectral_features(snv)
    hand_scaled = scaler.transform(hand_feat.reshape(1, -1))[0]
    
    # 3. Combine -> Shape (1, Features)
    X_full = np.concatenate([cnn_feat, hand_scaled]).reshape(1, -1)
    
    results = {}
    for target in TARGETS:
        # Global Predict
        rec = ensemble_models[target]
        X_sel = rec['selector'].transform(X_full)
        
        g_pred = 0
        for name, model in rec['models'].items():
            g_pred += rec['weights'][name] * model.predict(X_sel)[0]
            
        # Local Predict
        corr = correctors[target]
        sim = cosine_similarity(X_full, corr['X_train'])[0]
        nn_idx = np.argsort(sim)[-30:]
        
        lgb_m = lgb.LGBMRegressor(n_estimators=100, num_leaves=15, learning_rate=0.05, verbose=-1, n_jobs=1)
        lgb_m.fit(corr['X_train'][nn_idx], corr['residuals'][nn_idx])
        correction = lgb_m.predict(X_full)[0]
        
        # Merge
        final_val = g_pred + correction
        if target in LOG_TARGETS:
            final_val = np.expm1(final_val)
            
        # Round & Formatting
        if target in ['Fe', 'S']:
           results[target] = round(float(final_val), 2)
        else:
           results[target] = round(float(final_val), 3)

    return results

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not file.filename.endswith('.npz'):
        return {"error": "Only .npz files are supported."}
    
    try:
        content = await file.read()
        spectrum = process_npz(content)
        values = predict_soil(spectrum)
        
        # Prepare response
        resp = {}
        for target, val in values.items():
            resp[target] = {
                "value": val,
                "unit": UNITS[target]
            }
        return {"status": "success", "predictions": resp}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
