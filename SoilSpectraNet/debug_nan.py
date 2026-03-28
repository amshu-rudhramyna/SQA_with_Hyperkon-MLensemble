import torch
import numpy as np
from encoders import SpatialSpectral3DCNN, SpectralCNN, DomainDiscriminator, SpecBPPHead
from torch.utils.data import DataLoader
from phase5_domain_alignment import DomainDataset
from pathlib import Path

def debug_nan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    air_path = Path('data/train_hsi_pruned.npz')
    pri_path = Path('data/train_prisma.npz')
    
    d_air = np.load(air_path); d_pri = np.load(pri_path)
    ds = DomainDataset(d_air['X'], d_pri['X'])
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    
    air_enc = SpatialSpectral3DCNN(in_bands=512).to(device)
    pri_enc = SpectralCNN(in_bands=230).to(device)
    dann = DomainDiscriminator().to(device)
    bpp = SpecBPPHead().to(device)
    
    x_a = batch['x_air'].to(device)
    print(f"Input Airborne - Max: {x_a.max().item():.2f}, Min: {x_a.min().item():.2f}, HasNaN: {torch.isnan(x_a).any().item()}")
    
    # Trace Forward
    with torch.amp.autocast('cuda'):
        f_a = air_enc(x_a)
        print(f"Encoder Feature - Max: {f_a.max().item():.2f}, HasNaN: {torch.isnan(f_a).any().item()}")
        
        o_b = bpp(f_a)
        print(f"BPP Logit - Max: {o_b.max().item():.2f}, HasNaN: {torch.isnan(o_b).any().item()}")
        
        o_d = dann(f_a)
        print(f"DANN Logit - Max: {o_d.max().item():.2f}, HasNaN: {torch.isnan(o_d).any().item()}")
        
    print("Debug complete.")

if __name__ == '__main__':
    debug_nan()
