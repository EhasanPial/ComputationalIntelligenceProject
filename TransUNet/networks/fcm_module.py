# coding=utf-8
"""
Fuzzy C-Means (FCM) components for FCM-TransUNet.

Provides:
  - DifferentiableFCM  : soft membership U (B,N,C) and similarity S_f (B,N,N)
  - FCMSkipGate        : spatially gates ResNet skip connections with fuzzy maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable Fuzzy C-Means
# ---------------------------------------------------------------------------

class DifferentiableFCM(nn.Module):
    """
    Differentiable Fuzzy C-Means clustering on patch embeddings.

    Given token sequence x of shape (B, N, D):
      1. Initialises C cluster centres from a learnable linear projection
         of the mean-pooled token sequence.
      2. Iterates FCM membership and centre updates (no gradients blocked —
         fully differentiable via standard PyTorch ops).
      3. Returns:
           U   : (B, N, C)   per-token membership in each cluster
           S_f : (B, N, N)   fuzzy similarity  S_f = U @ U^T

    Attention bias formula:
        Attention_fuzzy = softmax( QK^T/sqrt(d) + λ·S_f ) V

    Args:
        num_clusters (int): number of fuzzy clusters C
        hidden_size  (int): token feature dimension D
        m            (float): fuzziness exponent (default 2.0)
        num_iters    (int): FCM update iterations (default 3; more = slower)
    """

    def __init__(self, num_clusters: int, hidden_size: int,
                 m: float = 2.0, num_iters: int = 3):
        super().__init__()
        self.C = num_clusters
        self.m = m
        self.num_iters = num_iters
        self.hidden_size = hidden_size

        # Projects mean-pooled tokens → C initial cluster centres
        self.center_init = nn.Linear(hidden_size, num_clusters * hidden_size)
        nn.init.xavier_uniform_(self.center_init.weight)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, N, D)  patch token embeddings

        Returns:
            U   : (B, N, C)  membership matrix  (rows sum to 1)
            S_f : (B, N, N)  fuzzy similarity matrix
        """
        B, N, D = x.shape
        C, m = self.C, self.m

        # ---- initialise cluster centres ----------------------------------
        x_mean = x.mean(dim=1)                            # (B, D)
        centers = self.center_init(x_mean).view(B, C, D)  # (B, C, D)

        # ---- iterative FCM updates ---------------------------------------
        for _ in range(self.num_iters):
            # Squared Euclidean distances  (B, N, C)
            x_exp = x.unsqueeze(2).expand(B, N, C, D)
            c_exp = centers.unsqueeze(1).expand(B, N, C, D)
            dist = (x_exp - c_exp).pow(2).sum(dim=-1).clamp(min=1e-10)  # (B,N,C)

            # Membership update
            # u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))
            exp = 2.0 / (m - 1.0)
            # dist ratio: (B, N, C, C) where [b,n,k,j] = dist[b,n,k]/dist[b,n,j]
            dist_ratio = dist.unsqueeze(3) / (dist.unsqueeze(2) + 1e-10)
            U = 1.0 / (dist_ratio.pow(exp).sum(dim=3) + 1e-10)  # (B, N, C)
            U = U / (U.sum(dim=2, keepdim=True) + 1e-10)         # row-normalise

            # Centre update
            # v_k = Σ_i u_ik^m · x_i  /  Σ_i u_ik^m
            U_m = U.pow(m)                                                  # (B,N,C)
            numerator   = torch.einsum('bnc,bnd->bcd', U_m, x)             # (B,C,D)
            denominator = U_m.sum(dim=1, keepdim=True).permute(0,2,1) + 1e-10  # (B,C,1)
            centers = numerator / denominator                               # (B,C,D)

        # ---- fuzzy similarity matrix  S_f = U @ U^T  (B, N, N) ----------
        S_f = torch.bmm(U, U.transpose(1, 2))

        return U, S_f


# ---------------------------------------------------------------------------
# FCM Skip Gate
# ---------------------------------------------------------------------------

class FCMSkipGate(nn.Module):
    """
    Spatially gates ResNet skip-connection features using fuzzy memberships.

    For each skip level the gate network:
      1. Reshapes U (B, N, K) → (B, K, H_f, W_f) spatial map
      2. Bilinearly up/downsamples to match the skip spatial size
      3. Projects K cluster channels → C_skip channels via 1×1 conv + Sigmoid
      4. Multiplies the skip feature map element-wise by the gate

    Args:
        skip_channels (int): number of channels in the skip feature map
        num_clusters  (int): number of fuzzy clusters (K)
    """

    def __init__(self, skip_channels: int, num_clusters: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(num_clusters, skip_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(skip_channels),
            nn.Sigmoid(),
        )
        # init gate bias to 1 so it starts as identity (open gate)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.ones_(self.gate[0].bias)

    def forward(self, skip: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skip : (B, C_skip, H, W)   ResNet skip feature map
            U    : (B, N, K)            fuzzy membership from bottleneck

        Returns:
            gated skip of shape (B, C_skip, H, W)
        """
        B, N, K = U.shape
        H, W = skip.shape[2], skip.shape[3]
        H_f = W_f = int(N ** 0.5)

        # Reshape to spatial
        U_spatial = U.transpose(1, 2).contiguous().view(B, K, H_f, W_f)

        # Resize to match skip dimensions if necessary
        if H_f != H or W_f != W:
            U_spatial = F.interpolate(
                U_spatial, size=(H, W), mode='bilinear', align_corners=False
            )

        gate_map = self.gate(U_spatial)   # (B, C_skip, H, W)
        return skip * gate_map
