import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, numpy as np, pickle
from torch.utils.data import DataLoader, TensorDataset

EPOCHS   = 350
LR       = 3e-4
BATCH    = 32
CNN_DIM  = 128
NOISE_STD = 0.005
DROP_RATE = 0.05
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
    def __init__(self, in_bands=150, embed_dim=CNN_DIM, n_targets=6, has_som=True):
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
        preds = {t: self.heads[t](emb).squeeze(-1) for t in self.heads}
        return emb, preds

class SpectralAugment:
    def __init__(self, noise_std=NOISE_STD, drop=DROP_RATE, training=True):
        self.noise_std = noise_std; self.drop = drop; self.training = training

    def __call__(self, x):
        if not self.training: return x
        x = x + torch.randn_like(x) * self.noise_std
        mask = (torch.rand(x.shape[-1]) > self.drop).float().to(x.device)
        return x * mask

def mtl_loss(preds, targets, weights=MTL_WEIGHTS):
    loss = 0.0
    for t, w in weights.items():
        if t in targets and targets[t] is not None:
            loss += w * nn.functional.mse_loss(preds[t], targets[t])
    return loss

if __name__ == '__main__':
    import os, pickle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_raw = torch.tensor(np.load('cache/X_raw.npy'))
    with open('cache/Y_log.pkl','rb') as f: Y_log = pickle.load(f)

    targets_t = {t: torch.tensor(Y_log[t]) for t in Y_log}
    targets_t['SOM'] = None

    bw = torch.tensor(np.load('cache/band_weights.npy'))
    model = SpectralCNN(in_bands=150).to(device)
    model.band_w.data = bw.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    aug = SpectralAugment()

    dataset = TensorDataset(X_raw, *[targets_t[t] if targets_t[t] is not None
                                      else torch.zeros(len(X_raw)) for t in MTL_WEIGHTS])
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    for epoch in range(EPOCHS):
        model.train(); aug.training = True; ep_loss = 0.0
        for batch in loader:
            xb = aug(batch[0].to(device))
            tb = {t: batch[i+1].to(device) for i,t in enumerate(MTL_WEIGHTS)}
            opt.zero_grad()
            _, preds = model(xb)
            loss = mtl_loss(preds, tb)
            loss.backward(); opt.step(); sched.step(epoch)
            ep_loss += loss.item()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  loss={ep_loss/len(loader):.4f}  lr={sched.get_last_lr()[0]:.2e}")

    os.makedirs('models/', exist_ok=True)
    torch.save(model.state_dict(), 'models/spectral_cnn.pt')
    print("CNN saved → models/spectral_cnn.pt")

