# coding=utf-8
"""
FCM-TransUNet  —  vit_seg_modeling.py
======================================
Extends vanilla TransUNet with:

  1. DifferentiableFCM  – runs on patch embeddings to produce:
       • U   (B, N, C)  membership matrix
       • S_f (B, N, N)  fuzzy similarity  S_f(i,j) = Σ_k u_ik·u_jk

  2. FuzzyAttention – MHSA with dual structural bias:
       Attn = softmax( QKᵀ/√d  +  λ·F  +  β·S ) V
         F_ij = Σ_k u_ik·u_jk          (fuzzy cluster similarity)
         S_ij = exp(-d(i,j)²/σ²)       (Gaussian spatial proximity)
       λ, β, σ are all per-layer learnable scalars.

  3. FCMSkipGate – wraps each ResNet skip connection with a
       spatial gate derived from the fuzzy memberships U.

Architecture (forward pass):
    Input (B,1,224,224)
      → repeat to RGB
      → ResNetV2 hybrid encoder  →  skip1..skip3  (gated by FCMSkipGate)
      → patch embeddings
      → DifferentiableFCM  →  U, S_f
      → Transformer encoder (FuzzyAttention layers receive S_f)
      → DecoderCup with gated skips
      → SegmentationHead  →  logits (B, n_classes, 224, 224)
"""

from __future__ import absolute_import, division, print_function

import copy
import logging
import math
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.nn import (CrossEntropyLoss, Dropout, LayerNorm,
                       Linear, Conv2d, Softmax)
from torch.nn.modules.utils import _pair

from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from .fcm_module import DifferentiableFCM, FCMSkipGate

logger = logging.getLogger(__name__)

# Weight-key constants for loading Google pre-trained checkpoints
ATTENTION_Q    = "MultiHeadDotProductAttention_1/query"
ATTENTION_K    = "MultiHeadDotProductAttention_1/key"
ATTENTION_V    = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT  = "MultiHeadDotProductAttention_1/out"
FC_0           = "MlpBlock_3/Dense_0"
FC_1           = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM       = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO → OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu":  torch.nn.functional.gelu,
    "relu":  torch.nn.functional.relu,
    "swish": swish,
}


# ===========================================================================
# Spatial Bias  S_ij = exp( -d(i,j)² / σ² )
# ===========================================================================

class SpatialBias(nn.Module):
    """
    Computes a Gaussian spatial proximity bias over the patch grid.

    For a sequence of N patches laid out on a √N × √N grid, the bias
    between patch i and patch j is:

        S_ij = exp( -d(i,j)² / σ² )

    where d(i,j) is the Euclidean distance between their 2-D grid
    coordinates (in patch units).

    Properties:
      • S_ii = 1   (a patch is maximally similar to itself)
      • Very close patches → ~1
      • Medium distance    → ~0.5  (when d ≈ σ·√ln2 ≈ 0.83σ)
      • Far patches        → ~0

    σ is stored as log(σ) so it remains strictly positive under gradient
    updates.  β (the global scale applied in attention) lives in
    FuzzyAttention, not here.

    The matrix is computed lazily on the first forward call for a given N
    and cached as a non-parameter buffer so it moves with the model to GPU.
    """

    def __init__(self, sigma_init: float = 7.0):
        super().__init__()
        # log-parameterised σ — always positive, freely optimisable
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma_init), dtype=torch.float32)
        )
        # cache: maps N → precomputed squared-distance matrix (N, N)
        self._dist2_cache: dict = {}

    # ------------------------------------------------------------------
    def _get_dist2(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Returns squared Euclidean distances on the √N × √N patch grid.
        Result shape: (N, N).  Cached after first call for each N.
        """
        if N not in self._dist2_cache:
            H = W = int(math.sqrt(N))
            # grid coordinates for every patch  (N, 2)
            rows = torch.arange(H, dtype=torch.float32, device=device)
            cols = torch.arange(W, dtype=torch.float32, device=device)
            # meshgrid → (H, W) each
            gy, gx = torch.meshgrid(rows, cols, indexing='ij')
            coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=1)  # (N,2)
            # pairwise squared distances  (N, N)
            diff   = coords.unsqueeze(0) - coords.unsqueeze(1)             # (N,N,2)
            dist2  = (diff ** 2).sum(dim=-1)                               # (N, N)
            self._dist2_cache[N] = dist2
        else:
            # move cached tensor to current device if necessary
            self._dist2_cache[N] = self._dist2_cache[N].to(device)

        return self._dist2_cache[N]

    # ------------------------------------------------------------------
    def forward(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            N      : number of patches (must be a perfect square)
            device : target device

        Returns:
            S : (N, N)  spatial bias matrix, values in (0, 1]
        """
        sigma  = self.log_sigma.exp()               # σ > 0  always
        dist2  = self._get_dist2(N, device)         # (N, N)
        S      = torch.exp(-dist2 / (sigma ** 2))   # (N, N)
        return S


# ===========================================================================
# Fuzzy + Spatial Dual-Bias Multi-Head Self-Attention
# ===========================================================================

class FuzzyAttention(nn.Module):
    """
    Multi-Head Self-Attention with two additive structural biases:

        scores = QKᵀ / √d  +  λ · F  +  β · S

    where:
        F  (B, N, N)  Fuzzy similarity:  F_ij = Σ_k u_ik · u_jk
                      Patches in the same fuzzy cluster attract each other.

        S  (N, N)     Spatial proximity:  S_ij = exp(-d(i,j)² / σ²)
                      Nearby patches attract each other regardless of content.

    Learnable scalars per transformer layer:
        λ   (fuzzy_lambda)  — scales fuzzy bias  (init 0.1)
        β   (spatial_beta)  — scales spatial bias (init 0.1)
        σ   (inside SpatialBias.log_sigma) — Gaussian bandwidth (init 7.0
            patch units ≈ half the 14×14 grid diagonal)

    When F is None the fuzzy term is skipped (plain MHSA + spatial bias).
    Spatial bias is always active once the layer is constructed.
    """

    def __init__(self, config, vis,
                 fuzzy_lambda_init:  float = 0.1,
                 spatial_beta_init:  float = 0.1,
                 spatial_sigma_init: float = 7.0):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key   = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out   = Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax      = Softmax(dim=-1)

        # ── Learnable bias scales ──────────────────────────────────────
        # λ · F  :  fuzzy cluster similarity
        self.fuzzy_lambda = nn.Parameter(
            torch.tensor(fuzzy_lambda_init, dtype=torch.float32)
        )
        # β · S  :  spatial Gaussian proximity
        self.spatial_beta = nn.Parameter(
            torch.tensor(spatial_beta_init, dtype=torch.float32)
        )

        # SpatialBias module owns log_σ and the dist² cache
        self.spatial_bias = SpatialBias(sigma_init=spatial_sigma_init)

    # ------------------------------------------------------------------
    def _transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)   # (B, h, N, d_h)

    # ------------------------------------------------------------------
    def forward(self, hidden_states, S_f=None):
        """
        Args:
            hidden_states : (B, N, D)
            S_f           : (B, N, N) fuzzy similarity F  [optional]

        Returns:
            attention_output : (B, N, D)
            weights          : attention probabilities or None
        """
        B, N, _ = hidden_states.shape

        Q = self._transpose_for_scores(self.query(hidden_states))
        K = self._transpose_for_scores(self.key(hidden_states))
        V = self._transpose_for_scores(self.value(hidden_states))

        # ── Base dot-product scores ────────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        # scores: (B, h, N, N)

        # ── λ · F  :  fuzzy bias ──────────────────────────────────────
        # F_ij ∈ [0,1]:  high when patches i,j share cluster membership
        if S_f is not None:
            scores = scores + self.fuzzy_lambda * S_f.unsqueeze(1)
            # S_f (B,N,N) → (B,1,N,N) broadcasts over heads

        # ── β · S  :  spatial bias ────────────────────────────────────
        # S_ij = exp(-d(i,j)²/σ²) ∈ (0,1]:  decays with patch distance
        S = self.spatial_bias(N, hidden_states.device)   # (N, N)
        scores = scores + self.spatial_beta * S.unsqueeze(0).unsqueeze(0)
        # S (N,N) → (1,1,N,N) broadcasts over batch and heads

        # ── Softmax + weighted sum ─────────────────────────────────────
        probs   = self.softmax(scores)
        weights = probs if self.vis else None
        probs   = self.attn_dropout(probs)

        ctx = torch.matmul(probs, V)                              # (B, h, N, d_h)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size()[:-2] + (self.all_head_size,))   # (B, N, D)

        out = self.proj_dropout(self.out(ctx))
        return out, weights

    # ------------------------------------------------------------------
    def load_from(self, weights, n_block):
        """Load pre-trained Q/K/V/out weights from Google ViT checkpoint.
        λ, β, and log_σ are new params — they keep their random-init values.
        """
        ROOT = f"Transformer/encoderblock_{n_block}"
        D    = self.all_head_size
        with torch.no_grad():
            q_w = np2th(weights[pjoin(ROOT, ATTENTION_Q,   "kernel")]).view(D, D).t()
            k_w = np2th(weights[pjoin(ROOT, ATTENTION_K,   "kernel")]).view(D, D).t()
            v_w = np2th(weights[pjoin(ROOT, ATTENTION_V,   "kernel")]).view(D, D).t()
            o_w = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(D, D).t()

            q_b = np2th(weights[pjoin(ROOT, ATTENTION_Q,   "bias")]).view(-1)
            k_b = np2th(weights[pjoin(ROOT, ATTENTION_K,   "bias")]).view(-1)
            v_b = np2th(weights[pjoin(ROOT, ATTENTION_V,   "bias")]).view(-1)
            o_b = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.query.weight.copy_(q_w); self.query.bias.copy_(q_b)
            self.key.weight.copy_(k_w);   self.key.bias.copy_(k_b)
            self.value.weight.copy_(v_w); self.value.bias.copy_(v_b)
            self.out.weight.copy_(o_w);   self.out.bias.copy_(o_b)


# ===========================================================================
# MLP (feed-forward) block
# ===========================================================================

class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1     = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2     = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn  = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act_fn(self.fc1(x)))))


# ===========================================================================
# Embeddings
# ===========================================================================

class Embeddings(nn.Module):
    """Patch + position embeddings (with optional ResNet hybrid backbone)."""

    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = None
        self.config  = config
        img_size     = _pair(img_size)

        if config.patches.get("grid") is not None:            # ResNet hybrid
            grid_size      = config.patches["grid"]
            patch_size      = (img_size[0] // 16 // grid_size[0],
                               img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches       = ((img_size[0] // patch_size_real[0]) *
                               (img_size[1] // patch_size_real[1]))
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches  = ((img_size[0] // patch_size[0]) *
                          (img_size[1] // patch_size[1]))
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(
            in_channels=in_channels, out_channels=config.hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)              # (B, D, √N, √N)
        x = x.flatten(2).transpose(-1, -2)        # (B, N, D)
        embeddings = self.dropout(x + self.position_embeddings)
        return embeddings, features


# ===========================================================================
# Transformer Block (passes S_f through)
# ===========================================================================

class Block(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.hidden_size    = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm       = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn            = Mlp(config)
        self.attn           = FuzzyAttention(config, vis)

    def forward(self, x, S_f=None):
        """
        Args:
            x   : (B, N, D)
            S_f : (B, N, N)  optional fuzzy bias

        Returns:
            x       : (B, N, D)
            weights : attention weights or None
        """
        h = x
        x, weights = self.attn(self.attention_norm(x), S_f=S_f)
        x = x + h

        h = x
        x = self.ffn(self.ffn_norm(x)) + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            self.attn.load_from(weights, n_block)

            mlp_w0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_w1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_b0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_b1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_w0); self.ffn.fc1.bias.copy_(mlp_b0)
            self.ffn.fc2.weight.copy_(mlp_w1); self.ffn.fc2.bias.copy_(mlp_b1)

            self.attention_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


# ===========================================================================
# Encoder
# ===========================================================================

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis          = vis
        self.layer        = nn.ModuleList(
            [copy.deepcopy(Block(config, vis))
             for _ in range(config.transformer["num_layers"])]
        )
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states, S_f=None):
        """
        All layers receive the same S_f computed once at the bottleneck.
        """
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states, S_f=S_f)
            if self.vis:
                attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights


# ===========================================================================
# Transformer (adds FCM after embeddings)
# ===========================================================================

class Transformer(nn.Module):
    """
    Wraps Embeddings + DifferentiableFCM + Encoder.

    After patch embeddings are formed, FCM is applied to produce U and S_f.
    S_f is broadcast to all encoder layers as an attention bias.
    U is returned so the decoder can apply FCMSkipGates.
    """

    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder    = Encoder(config, vis)

        # FCM parameters read from config (with safe defaults)
        num_clusters = getattr(config, 'fcm_clusters',    8)
        fcm_iters    = getattr(config, 'fcm_iters',       3)
        fcm_m        = getattr(config, 'fcm_m',         2.0)

        self.fcm = DifferentiableFCM(
            num_clusters=num_clusters,
            hidden_size=config.hidden_size,
            m=fcm_m,
            num_iters=fcm_iters,
        )

    def forward(self, input_ids):
        embeddings, features = self.embeddings(input_ids)   # (B,N,D), list|None

        # ---- FCM on patch embeddings ------------------------------------
        U, S_f = self.fcm(embeddings)   # U:(B,N,K)  S_f:(B,N,N)

        # ---- Transformer encoder with fuzzy bias ------------------------
        encoded, attn_weights = self.encoder(embeddings, S_f=S_f)

        return encoded, attn_weights, features, U


# ===========================================================================
# Decoder helpers
# ===========================================================================

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding,
                         bias=not use_batchnorm)
        bn   = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels,
                                kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels,
                                kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.up    = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                           padding=kernel_size // 2)
        up = (nn.UpsamplingBilinear2d(scale_factor=upsampling)
              if upsampling > 1 else nn.Identity())
        super().__init__(conv2d, up)


# ===========================================================================
# Decoder with FCMSkipGates
# ===========================================================================

class DecoderCup(nn.Module):
    """
    Standard TransUNet decoder, extended with per-level FCMSkipGates.

    For each skip level i that has non-zero skip channels, an FCMSkipGate
    gates the ResNet feature map using the fuzzy membership map U before
    concatenation.
    """

    def __init__(self, config):
        super().__init__()
        self.config       = config
        head_channels     = 512
        num_clusters      = getattr(config, 'fcm_clusters', 8)

        self.conv_more = Conv2dReLU(config.hidden_size, head_channels,
                                    kernel_size=3, padding=1)

        decoder_channels = config.decoder_channels
        in_channels  = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = list(self.config.skip_channels)        # mutable copy
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        self.skip_channels = skip_channels

        self.blocks = nn.ModuleList([
            DecoderBlock(ic, oc, sc)
            for ic, oc, sc in zip(in_channels, out_channels, skip_channels)
        ])

        # One FCMSkipGate per active skip level
        self.fcm_skip_gates = nn.ModuleList([
            FCMSkipGate(skip_ch, num_clusters) if skip_ch > 0 else nn.Identity()
            for skip_ch in skip_channels
        ])

    def forward(self, hidden_states, features=None, U=None):
        """
        Args:
            hidden_states : (B, N, D)  transformer output
            features      : list of ResNet skip tensors  (or None)
            U             : (B, N, K)  fuzzy membership from FCM  (or None)
        """
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        for i, (decoder_block, gate) in enumerate(
                zip(self.blocks, self.fcm_skip_gates)):

            if features is not None and i < self.config.n_skip:
                skip = features[i]
                # Apply FCMSkipGate when U is available and this level is active
                if U is not None and self.skip_channels[i] > 0:
                    skip = gate(skip, U)
            else:
                skip = None

            x = decoder_block(x, skip=skip)

        return x


# ===========================================================================
# Top-level VisionTransformer (FCM-TransUNet)
# ===========================================================================

class VisionTransformer(nn.Module):
    """
    FCM-TransUNet.

    Drop-in replacement for the original TransUNet VisionTransformer.
    Adds:
      • DifferentiableFCM on patch tokens (inside Transformer)
      • FuzzyAttention in every transformer block
      • FCMSkipGate on every active decoder skip connection
    """

    def __init__(self, config, img_size=224, num_classes=21843,
                 zero_head=False, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head   = zero_head
        self.classifier  = config.classifier
        self.config      = config

        self.transformer = Transformer(config, img_size, vis)
        self.decoder     = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )

    # ------------------------------------------------------------------
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Transformer: returns encoded tokens, attention weights,
        #              ResNet skip features, and fuzzy membership U
        x, attn_weights, features, U = self.transformer(x)

        # Decoder with gated skips
        x = self.decoder(x, features=features, U=U)

        logits = self.segmentation_head(x)
        return logits

    # ------------------------------------------------------------------
    def load_from(self, weights):
        """Load Google ViT pre-trained weights (patch embed + transformer)."""
        with torch.no_grad():
            res_weight = weights

            # Patch embeddings
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"]))

            # Encoder norms
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))

            # Position embeddings (with resize if needed)
            posemb     = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings

            if posemb.size() == posemb_new.size():
                posemb_new.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb_new.copy_(posemb[:, 1:])
            else:
                logger.info("load_pretrained: resized variant: %s to %s"
                            % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom_factors = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid  = ndimage.zoom(posemb_grid, zoom_factors, order=1)
                posemb_grid  = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb_new.copy_(np2th(posemb_grid))

            # Encoder blocks
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            # ResNet hybrid weights
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True))
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(
                    np2th(res_weight["gn_root/scale"]).view(-1))
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(
                    np2th(res_weight["gn_root/bias"]).view(-1))

                for bname, block in (self.transformer.embeddings
                                     .hybrid_model.body.named_children()):
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


# ===========================================================================
# Config registry
# ===========================================================================

CONFIGS = {
    'ViT-B_16':     configs.get_b16_config(),
    'ViT-B_32':     configs.get_b32_config(),
    'ViT-L_16':     configs.get_l16_config(),
    'ViT-L_32':     configs.get_l32_config(),
    'ViT-H_14':     configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing':      configs.get_testing(),
}