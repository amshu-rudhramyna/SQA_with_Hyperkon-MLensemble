import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

class ShallowSpectralCNN1D(nn.Module):
    """
    Step A2: Shallow 1D CNN for local spectral window feature extraction.
    Captures neighborhood correlations without over-parameterization.
    Input: (B, 1, Bands)
    """
    def __init__(self, in_bands=1290, out_dim=64):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):
        # x: (B, Bands) or (B, 1, Bands)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            # Mean spatial pooling if input is (B, C, H, W)
            x = torch.mean(x, dim=(2, 3))
            if len(x.shape) == 2: x = x.unsqueeze(1)
            
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SpatialSpectral3DCNN(nn.Module):
    """
    Approach 2: 3D CNN (Spectral + Spatial)
    Processes (B, 1, Bands, H, W) natively learning volumetric correlations.
    """
    def __init__(self, in_bands=512, out_dim=128):
        super().__init__()
        # Optimized 3D Conv: reducing the pruned 512 bands to a 128 latent space
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(4, 2, 2))
        
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(4, 2, 2))
        
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # Adaptive pool to fixed representation regardless of padding sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x is originally (B, Bands, H, W). Add channel dim -> (B, 1, Bands, H, W)
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
            
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SpectralCNN(nn.Module):
    """
    2D CNN kept for processing shallow MSI satellite bands natively (B, 12, 8, 8).
    """
    def __init__(self, in_bands, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_bands, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x is (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pool(x).view(x.size(0), -1)
        return x

class CrossAttentionFusion(nn.Module):
    """
    Approach 2: Cross-Attention Fusion
    HSI features act as Queries, MSI as Keys/Values.
    """
    def __init__(self, hsi_dim=128, msi_dim=64, embed_dim=128, num_heads=4):
        super().__init__()
        # Project source dimensions to the common attention space
        self.q_proj = nn.Linear(hsi_dim, embed_dim)
        self.k_proj = nn.Linear(msi_dim, embed_dim)
        self.v_proj = nn.Linear(msi_dim, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, hsi_feat, msi_feat):
        # inputs are (B, Dim). We add sequence length = 1 for MultiHeadAttention: (B, 1, Dim)
        Q = self.q_proj(hsi_feat).unsqueeze(1)
        K = self.k_proj(msi_feat).unsqueeze(1)
        V = self.v_proj(msi_feat).unsqueeze(1)
        
        # Cross Attention
        attn_out, _ = self.attention(Q, K, V)
        
        # Residual + Norm
        x = self.norm1(Q + attn_out)
        
        # FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Return to (B, embed_dim)
        return x.squeeze(1)

# ===================================================================
# PHASE 3: DOMAIN ALIGNMENT (DANN) & SpecBPP
# ===================================================================

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class DomainDiscriminator(nn.Module):
    """
    Classifies if input features come from Airborne (0) or Satellite (1).
    Used with GRL to achieve sensor-invariant representation.
    """
    def __init__(self, input_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.net(x)

class SpecBPPHead(nn.Module):
    """
    Predicts if band order is permutation-correct (0) or shuffled (1).
    (Phase 2.2C: Best Representation Strategy)
    """
    def __init__(self, input_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def augment_view_3d(x, noise_std=0.03, mask_prob=0.15):
    """
    Pretraining augmentations compatible with (B, Bands, H, W)
    """
    torch.backends.cudnn.benchmark = True # Global speed optimization
    # Fix for potentially zero-valued mask causing division by zero in normalize
    out = x + torch.randn_like(x) * noise_std
    mask = (torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device) > mask_prob).float()
    return out * mask

def contrastive_loss(z1, z2, temperature=0.1):
    # Stabilized normalization
    z1 = F.normalize(z1, dim=1, eps=1e-8)
    z2 = F.normalize(z2, dim=1, eps=1e-8)
    
    # Compute logits and clip to prevent gradient explosion in cross_entropy
    logits = torch.matmul(z1, z2.T) / temperature
    # Clipping logits to a reasonable range for numerical stability in Softmax
    logits = torch.clamp(logits, -50.0, 50.0)
    
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def pretrain_encoder(encoder, dataloader, device, epochs=15):
    """
    Trains the 3D encoder using SimCLR style contrastive loss leveraging RTX 3070 Ti.
    """
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=5e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    encoder.to(device)
    encoder.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch_x = batch[0].to(device)
            optimizer.zero_grad()
            
            view1 = augment_view_3d(batch_x)
            view2 = augment_view_3d(batch_x)
            
            with torch.amp.autocast('cuda'):
                z1 = encoder(view1)
                z2 = encoder(view2)
                loss = contrastive_loss(z1, z2)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
        print(f"Pretraining Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    return encoder

# ===================================================================
# PHASE 4: SPECTRAL-SPATIAL TRANSFORMER (SST) - THE "BEST" OPTION
# ===================================================================

class SpectralSpatialTransformer(nn.Module):
    """
    Component 3 - Approach 3: Spectral-Spatial Transformer
    Treats the HSI patch (B, Bands, 20, 20) as a sequence of Spatial Patches.
    """
    def __init__(self, in_bands=1290, embed_dim=128, patch_size=4, num_layers=4, num_heads=8, dropout=0.2, out_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.in_bands = in_bands
        
        # 1. Spectral Bottleneck: Maintain more spectral resolution
        # Using 1x1 Conv3d with stride on bands to reduce compute
        self.spectral_proj = nn.Conv3d(1, 16, kernel_size=(15, 1, 1), stride=(4, 1, 1), padding=(7, 0, 0))
        # Adaptive pool to fixed 64 "super-bands" instead of 128 to save memory/speed
        self.spectral_pool = nn.AdaptiveAvgPool3d((64, 20, 20)) 
        
        # 2. Patch Embedding
        # 20x20 grid with 4x4 patches -> 5x5 patches = 25 tokens
        # Spectral dimension after pool is 16 (channels) * 64 (bands) = 1024
        num_patches = (20 // patch_size) ** 2
        patch_dim = (16 * 64) * patch_size * patch_size # 1024 * 16 = 16383
        
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, Bands, 20, 20)
        if len(x.shape) == 4:
            x = x.unsqueeze(1) # (B, 1, Bands, 20, 20)
            
        # Spectral Reduction
        x = F.relu(self.spectral_proj(x))
        x = self.spectral_pool(x) # (B, 32, 128, 20, 20) -> (B, 32, 128, 20, 20) if pool correct
        # Wait, AdaptiveAvgPool3d((128, 20, 20)) would reduce BANDS to 128.
        # Shape: (B, 32, 128, 20, 20)
        
        # Flatten and Patchify
        B, C, Bands, H, W = x.shape
        # Flatten C into Bands dim for easier patching: (B, C*Bands, H, W) -> (B, 32*128, 20, 20)
        x = x.view(B, -1, H, W)
        
        # (B, Dim, H, W) -> (B, Num_Patches, Patch_Dim)
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p) # (B, Dim, 5, 5, 4, 4)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # (B, 5, 5, Dim, 4, 4)
        patches = patches.view(B, 25, -1) # (B, 25, Dim*p*p)
        
        # Embed
        x = self.patch_to_embedding(patches) # (B, 25, embed_dim)
        
        # Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 26, embed_dim)
        
        # Positional Encoding
        x += self.pos_embedding
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Return CLS Token feature
        return self.mlp_head(x[:, 0])
