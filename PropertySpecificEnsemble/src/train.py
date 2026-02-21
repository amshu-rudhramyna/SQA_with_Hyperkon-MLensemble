import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure the root directory is in the python path to find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hyperkon import HyperKon
from src.data.loader import HyperviewDataset
from torch.utils.data import DataLoader

class SpecifcTraceMTLLoss(nn.Module):
    def __init__(self, l1_weight=0.1, device='cuda'):
        """
        Multi-Task Learning Loss explicitly tailored for the HYPERVIEW2 Trace Elements.
        Boron (B) and Copper (Cu) are weighted heavily because they lack standard visible-range markers.
        Fe and Mn are weighted neutrally.
        """
        super(SpecifcTraceMTLLoss, self).__init__()
        self.l1_weight = l1_weight
        self.device = device
        
        # Target Map: B, Cu, Zn, Fe, S, Mn
        # Weight emphasis on Boron (3.0) and Copper (2.5) to combat severe latency and low concentration
        # Zinc (1.5), Sulfur (1.5), Iron (1.0), Mn (1.0)
        self.prop_weights = torch.tensor([3.0, 2.5, 1.5, 1.0, 1.5, 1.0], dtype=torch.float32).to(device)

    def forward(self, preds, targets):
        mse = (preds - targets) ** 2
        weighted_mse = mse * self.prop_weights
        
        l1 = torch.abs(preds - targets)
        weighted_l1 = l1 * self.prop_weights
        
        return weighted_mse.mean() + self.l1_weight * weighted_l1.mean()

def train_hyperkon_phase1(data_dir, epochs=100, batch_size=24, lr=1e-3, min_lr=1e-6, device='cuda'):
    print("Initializing Tier 2: Automatic Mixed Precision (AMP) CNN Fine-Tuning...")
    
    # B, Cu, Zn, Fe, S, Mn
    num_targets = 6
    model = HyperKon(num_features=num_targets).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    criterion = SpecifcTraceMTLLoss(device=device)

    dataset = HyperviewDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize the AMP feature scaler to prevent float16 underflow during training
    scaler = torch.amp.GradScaler('cuda' if device == 'cuda' else 'cpu', enabled=(device=='cuda'))
    
    print(f"Dataset Size: {len(dataset)} | AMP Batch Size: {batch_size}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            inputs = batch['cnn_input'].to(device)
            targets = batch['targets'].to(device)
            
            if targets.shape[1] > num_targets:
                targets = targets[:, :num_targets]
            
            optimizer.zero_grad()
            
            # Autocast enables Mixed Precision
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', enabled=(device=='cuda')):
                preds = model(inputs)
                loss = criterion(preds, targets)
            
            # Scaler scales the loss, executes backward pass, and un-scales optimizers
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        print(f"Epoch {epoch} Average Loss: {epoch_loss / len(dataloader):.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/hyperkon_phase1_trace.pth')
    print("Tier 2 Encoder Phase Complete. Model saved to checkpoints/hyperkon_phase1_trace.pth")
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data is expected globally at the root
    train_hyperkon_phase1(data_dir='../data/raw/HYPERVIEW2/train', device=device)
    print(f"AMP Trainer execution successfully finalized on: {device}")
