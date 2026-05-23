import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """3-block CNN. forward() returns (logits, [z1, z2, z3]) where z_j is
    the post-ReLU feature map before pooling. K_CNN = 3."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1: (B,3,32,32) -> relu -> z1 (B,32,16,16) -> pool
        h = self.relu(self.bn1(self.conv1(x)))
        z1 = h                          # (B, 32, 32, 32) before pool
        h = self.pool(h)                # (B, 32, 16, 16)

        # Block 2: (B,32,16,16) -> relu -> z2 (B,64,8,8) -> pool
        h = self.relu(self.bn2(self.conv2(h)))
        z2 = h                          # (B, 64, 16, 16) before pool
        h = self.pool(h)                # (B, 64, 8, 8)

        # Block 3: (B,64,8,8) -> relu -> z3 (B,128,4,4) -> pool
        h = self.relu(self.bn3(self.conv3(h)))
        z3 = h                          # (B, 128, 8, 8) before pool
        h = self.pool(h)                # (B, 128, 4, 4)

        h = self.gap(h).flatten(1)      # (B, 128)
        logits = self.fc(h)
        return logits, [z1, z2, z3]


class SmallRNN(nn.Module):
    """2-layer GRU over row sequences. forward() returns (logits, [z1,z2,z3,z4])
    where z_k is the hidden state at steps 7, 15, 23, 31. K_RNN = 4."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size=96, hidden_size=128, num_layers=2,
                          batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        B = x.size(0)
        x_seq = x.view(B, 32, 96)          # (B, T=32, 96)
        out, _ = self.gru(x_seq)            # (B, 32, 128)
        z1 = out[:, 7, :]
        z2 = out[:, 15, :]
        z3 = out[:, 23, :]
        z4 = out[:, 31, :]
        logits = self.fc(z4)
        return logits, [z1, z2, z3, z4]
