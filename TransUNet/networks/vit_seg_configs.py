# coding=utf-8
"""
vit_seg_configs.py  —  FCM-TransUNet edition
============================================
Adds three FCM hyperparameters to every config:

  fcm_clusters  (int)   number of fuzzy clusters C          [default: 8]
  fcm_iters     (int)   FCM update iterations               [default: 3]
  fcm_m         (float) fuzziness exponent  m > 1           [default: 2.0]

These are read by Transformer.__init__() with safe get-defaults,
so old checkpoints trained without FCM params will still load fine.
"""

import ml_collections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_fcm_defaults(config,
                      fcm_clusters: int   = 8,
                      fcm_iters:    int   = 3,
                      fcm_m:        float = 2.0):
    """Attach FCM hyperparameters to an existing config dict."""
    config.fcm_clusters = fcm_clusters
    config.fcm_iters    = fcm_iters
    config.fcm_m        = fcm_m
    return config


# ---------------------------------------------------------------------------
# ViT-B/16
# ---------------------------------------------------------------------------

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return _add_fcm_defaults(config)


# ---------------------------------------------------------------------------
# Minimal test config
# ---------------------------------------------------------------------------

def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return _add_fcm_defaults(config, fcm_clusters=2, fcm_iters=1)


# ---------------------------------------------------------------------------
# R50 + ViT-B/16  (primary model for Synapse)
# ---------------------------------------------------------------------------

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    # FCM: 8 clusters matches well with the 9 organ classes on Synapse
    return _add_fcm_defaults(config, fcm_clusters=8, fcm_iters=3, fcm_m=2.0)


# ---------------------------------------------------------------------------
# ViT-B/32
# ---------------------------------------------------------------------------

def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config   # fcm defaults already set by get_b16_config


# ---------------------------------------------------------------------------
# ViT-L/16
# ---------------------------------------------------------------------------

def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return _add_fcm_defaults(config)


# ---------------------------------------------------------------------------
# R50 + ViT-L/16
# ---------------------------------------------------------------------------

def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration."""
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return _add_fcm_defaults(config)


# ---------------------------------------------------------------------------
# ViT-L/32
# ---------------------------------------------------------------------------

def get_l32_config():
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


# ---------------------------------------------------------------------------
# ViT-H/14
# ---------------------------------------------------------------------------

def get_h14_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return _add_fcm_defaults(config)
