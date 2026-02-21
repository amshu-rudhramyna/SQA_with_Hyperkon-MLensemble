import os
import sys
import torch
import torch.nn as nn

# Ensure the root directory is in the python path to find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm
import os

from src.models.hyperkon import HyperKon
from src.data.loader import HyperviewDataset
from torch.utils.data import DataLoader

class MultiTaskLoss(nn.Module):
    def __init__(self, l1_weight=0.1, prop_weights=None, device='cuda'):
        super(MultiTaskLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l1 = nn.L1Loss(reduction='none') # Don't mean yet to apply weights
        self.device = device
        
        # B: 1.5, Cu: 1.2, Zn: 1.2, Fe: 1.0, S: 1.2, Mn: 1.0
        if prop_weights is None:
            self.prop_weights = torch.tensor([1.5, 1.2, 1.2, 1.0, 1.2, 1.0], dtype=torch.float32).to(device)
        else:
            self.prop_weights = torch.tensor(prop_weights, dtype=torch.float32).to(device)

    def forward(self, preds, targets):
        mse = (preds - targets) ** 2
        weighted_mse = mse * self.prop_weights
        
        l1 = torch.abs(preds - targets)
        weighted_l1 = l1 * self.prop_weights
        
        return weighted_mse.mean() + self.l1_weight * weighted_l1.mean()

def train_hyperkon_phase1(data_dir, epochs=100, batch_size=24, lr=1e-3, min_lr=1e-6, device='cuda'):
    print("Initializing Phase 1: CNN Fine-Tuning...")
    
    # Needs to be updated depending on actual target dimensions
    # Actual dataset uses 6 targets: B, Cu, Zn, Fe, S, Mn
    num_targets = 6
    model = HyperKon(num_features=num_targets).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    criterion = MultiTaskLoss(device=device)

    dataset = HyperviewDataset(data_dir)
    # Using small dataloader workers for windows
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Dataset Size: {len(dataset)} | Target Batch Size: {batch_size}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # tqdm for progress
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            inputs = batch['cnn_input'].to(device)
            targets = batch['targets'].to(device)
            
            # targets from loader might include more or fewer columns, adjust indexing.
            # Assuming first num_targets are the properties we want
            if targets.shape[1] > num_targets:
                targets = targets[:, :num_targets]
            
            optimizer.zero_grad()
            preds = model(inputs)
            
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        print(f"Epoch {epoch} Average Loss: {epoch_loss / len(dataloader):.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/hyperkon_phase1.pth')
    print("Phase 1 Complete. Model saved to checkpoints/hyperkon_phase1.pth")
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_hyperkon_phase1(data_dir='../data/raw/HYPERVIEW2/train', device=device)
    print(f"Trainer ready using device: {device}")
