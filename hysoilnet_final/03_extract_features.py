import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np, pickle

CNN_DIM = 128

def extract_embeddings(model, X_raw, device, batch=64):
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_raw), batch):
            xb = torch.tensor(X_raw[i:i+batch]).to(device)
            emb, _ = model(xb)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)

if __name__ == '__main__':
    train_cnn = __import__('02_train_cnn')
    SpectralCNN = train_cnn.SpectralCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_raw  = np.load('cache/X_raw.npy')
    X_feat = np.load('cache/X_feat.npy')

    model = SpectralCNN().to(device)
    model.load_state_dict(torch.load('models/spectral_cnn.pt', map_location=device))
    emb = extract_embeddings(model, X_raw, device)

    X_full = np.concatenate([emb, X_feat], axis=1)
    np.save('cache/X_full.npy', X_full.astype(np.float32))
    print(f"Full feature matrix: {X_full.shape}")

