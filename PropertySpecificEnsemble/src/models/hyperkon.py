import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SpectralAttention1D(nn.Module):
    """
    Applies attention across the sequence (spectral length) dimension.
    """
    def __init__(self, in_channels):
        super(SpectralAttention1D, self).__init__()
        # 1D conv to get attention weights across the sequence length
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is (B, C, L)
        attn = self.conv(x) # (B, 1, L)
        attn = self.sigmoid(attn)
        return x * attn

class ResNeXtBottleneck1D(nn.Module):
    """
    ResNeXt 1D Bottleneck block with SE module.
    """
    expansion = 2 # Usually 4, but paper target ~5.54M. Adjusting to keep parameter count manageable.

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, downsample=None):
        super(ResNeXtBottleneck1D, self).__init__()
        D = int(out_channels * (base_width / 64.0))
        group_width = cardinality * D
        
        self.conv1 = nn.Conv1d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(group_width)
        
        self.conv2 = nn.Conv1d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm1d(group_width)
        
        self.conv3 = nn.Conv1d(group_width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.se = SELayer1D(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HyperKon(nn.Module):
    """
    HyperKon: ResNeXt-based 1D-CNN Backbone
    """
    def __init__(self, block=ResNeXtBottleneck1D, layers=[3, 4, 6, 3], cardinality=32, base_width=4, num_features=128):
        super(HyperKon, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.base_width = base_width

        # Initial layer
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Spectral Attention after initial conv
        self.spectral_attn = SpectralAttention1D(self.in_channels)

        # ResNeXt Stages
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        # Global Context Module (GAP + GMP)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Output embedding: Concat GAP and GMP (256*expansion * 2) -> 128
        in_features = 256 * block.expansion * 2
        
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_features)
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.cardinality, self.base_width, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, self.cardinality, self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x expected shape: (Batch, 1, 150)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.spectral_attn(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Context
        gap = self.gap(x).view(x.size(0), -1)
        gmp = self.gmp(x).view(x.size(0), -1)
        x_ctx = torch.cat((gap, gmp), dim=1)

        # Embedding
        emb = self.embedding(x_ctx)
        
        return emb

if __name__ == "__main__":
    # Test network
    model = HyperKon(base_width=8)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HyperKon Parameter Count: {total_params / 1e6:.2f}M")
    
    # Dummy input: Batch=2, Channels=1, Length=150
    dummy_input = torch.randn(2, 1, 150)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")  # Should be (2, 128)
