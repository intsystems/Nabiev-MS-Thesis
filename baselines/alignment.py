import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftAlignment(nn.Module):
    """Computes soft alignment matrix Gamma from teacher and student PSD matrices.

    direction='row': teacher-to-student, rows sum to 1 (each teacher layer
                     matched to a mixture of student layers).
    direction='col': student-to-teacher, cols sum to 1 (each student layer
                     matched to a mixture of teacher layers).
    """

    def __init__(self, tau=1.0, direction='row'):
        super().__init__()
        self.tau = tau
        self.direction = direction

    def forward(self, F_T_list, F_S_list):
        KT = len(F_T_list)
        KS = len(F_S_list)
        B = F_T_list[0].shape[0]
        device = F_T_list[0].device

        C = torch.zeros(B, KT, KS, device=device)
        for k in range(KT):
            for j in range(KS):
                diff = F_T_list[k] - F_S_list[j]           # (B, d, d)
                C[:, k, j] = (diff ** 2).sum(dim=(1, 2))   # (B,)

        if self.direction == 'row':
            Gamma = F.softmax(-C / self.tau, dim=2)         # rows sum to 1
        else:
            Gamma = F.softmax(-C / self.tau, dim=1)         # cols sum to 1

        return Gamma, C


def distillation_loss(Gamma, C):
    """Weighted sum of pairwise Frobenius costs. Scalar."""
    return (Gamma * C).sum(dim=(1, 2)).mean()


def gamma_entropy(Gamma, direction='row'):
    """Mean entropy of the alignment distribution over the normalization axis."""
    eps = 1e-8
    if direction == 'row':
        # Gamma: (B, KT, KS), normalize over KS (dim=2)
        H = -(Gamma * (Gamma + eps).log()).sum(dim=2)   # (B, KT)
        return H.mean().item()
    else:
        # normalize over KT (dim=1)
        H = -(Gamma * (Gamma + eps).log()).sum(dim=1)   # (B, KS)
        return H.mean().item()
