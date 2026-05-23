import torch
import torch.nn as nn


class CNNProjector(nn.Module):
    """Maps a CNN feature map (B, D2, H, W) to a (B, d, d) normalized Gram matrix."""

    def __init__(self, in_channels, d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d, kernel_size=1, bias=False)

    def forward(self, A):
        # A: (B, D2, H, W)
        B = A.size(0)
        A_tilde = self.conv(A)                              # (B, d, H, W)
        d, H, W = A_tilde.shape[1], A_tilde.shape[2], A_tilde.shape[3]
        P = H * W
        B_mat = A_tilde.view(B, d, P)                       # (B, d, P)
        F = (B_mat @ B_mat.transpose(1, 2)) / P             # (B, d, d)
        return F


class RNNProjector(nn.Module):
    """Maps an RNN hidden state (B, in_dim) to a (B, d, d) rank-1 PSD matrix."""

    def __init__(self, in_dim, d):
        super().__init__()
        self.linear = nn.Linear(in_dim, d)

    def forward(self, h):
        # h: (B, in_dim)
        u = self.linear(h)                                  # (B, d)
        F = u.unsqueeze(2) @ u.unsqueeze(1)                 # (B, d, d)
        return F


def build_projectors(teacher_arch, student_arch, d=32):
    """Return (teacher_projectors, student_projectors) as nn.ModuleList.

    CNN layers expose feature maps with channels [32, 64, 128].
    RNN layers expose hidden states with dim 128.
    """
    cnn_channels = [32, 64, 128]
    rnn_dim = 128

    def make_proj_list(arch):
        if arch == 'cnn':
            return nn.ModuleList([CNNProjector(c, d) for c in cnn_channels])
        else:
            return nn.ModuleList([RNNProjector(rnn_dim, d) for _ in range(4)])

    return make_proj_list(teacher_arch), make_proj_list(student_arch)
