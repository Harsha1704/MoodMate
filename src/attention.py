"""
attention.py  ── CBAM Attention Module (standalone)
═════════════════════════════════════════════════════
Convolutional Block Attention Module (CBAM)
Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

This file can be imported independently from train_model.py:

    from src.attention import CBAM, ChannelAttention, SpatialAttention

Why attention helps for FER2013:
  • Faces contain highly localised expression cues (eyes, brow, mouth, jaw).
  • Channel attention learns WHICH feature detectors matter (e.g. edge
    detectors for wrinkles, texture channels for lip corners).
  • Spatial attention learns WHERE to look — suppresses background/hair
    and amplifies the eye and mouth regions.
  • Together they give the model an implicit "look here" mechanism that
    directly improves recognition of subtle emotions (fear, disgust).
"""

import tensorflow as tf
from tensorflow.keras import layers


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL ATTENTION  (Squeeze-and-Excitation style)
# ══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(layers.Layer):
    """
    Channel attention gate.

    Learns a per-channel importance weight by:
      1. Squeeze: Global Average Pool + Global Max Pool → (B, 1, 1, C) each
      2. Excitation: shared MLP (C → C/r → C) on both pooled descriptors
      3. Merge: element-wise sum → sigmoid → scale input feature map

    Args:
        channels : Number of input feature map channels.
        r        : Reduction ratio for the bottleneck MLP (default 8).
                   Smaller r = more capacity but more params.

    Input / Output shapes:
        (B, H, W, C)  →  (B, H, W, C)   [same shape, channel-scaled]
    """
    def __init__(self, channels: int, r: int = 8, **kwargs):
        super().__init__(**kwargs)
        hidden = max(channels // r, 8)      # never go below 8 neurons
        self.gap  = layers.GlobalAveragePooling2D(keepdims=True)  # (B,1,1,C)
        self.gmp  = layers.GlobalMaxPooling2D(keepdims=True)      # (B,1,1,C)
        self.fc1  = layers.Dense(hidden, activation="relu",  use_bias=False)
        self.fc2  = layers.Dense(channels,                   use_bias=False)

    def call(self, x, training=None):
        avg_pool  = self.fc2(self.fc1(self.gap(x)))   # (B, 1, 1, C)
        max_pool  = self.fc2(self.fc1(self.gmp(x)))   # (B, 1, 1, C)
        gate      = tf.nn.sigmoid(avg_pool + max_pool) # (B, 1, 1, C)
        return x * gate

    def get_config(self):
        cfg = super().get_config()
        cfg.update(channels=self.fc2.units, r=8)
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class SpatialAttention(layers.Layer):
    """
    Spatial attention gate.

    Learns a per-pixel (spatial) importance map by:
      1. Pool across channels: avg + max → (B, H, W, 2)
      2. Conv 7×7 (or kernel_size) → (B, H, W, 1)
      3. Sigmoid → spatial mask
      4. Scale input feature map

    Args:
        kernel_size : Convolution kernel for the spatial descriptor (default 7).
                      Larger kernel captures broader spatial context.

    Input / Output shapes:
        (B, H, W, C)  →  (B, H, W, C)   [same shape, spatially-scaled]
    """
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name="spatial_conv",
        )

    def call(self, x, training=None):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B, H, W, 1)
        max_pool = tf.reduce_max(x,  axis=-1, keepdims=True)  # (B, H, W, 1)
        concat   = tf.concat([avg_pool, max_pool], axis=-1)   # (B, H, W, 2)
        gate     = self.conv(concat)                           # (B, H, W, 1)
        return x * gate

    def get_config(self):
        cfg = super().get_config()
        cfg.update(kernel_size=self.kernel_size)
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# CBAM  (Channel → Spatial, sequential)
# ══════════════════════════════════════════════════════════════════════════════

class CBAM(layers.Layer):
    """
    Full CBAM block: Channel Attention → Spatial Attention (sequential).

    Reference: Woo et al., ECCV 2018. https://arxiv.org/abs/1807.06521

    Usage in a model:
        x = backbone(inputs)           # (B, H, W, C)
        x = CBAM(channels=C)(x)        # attended feature map
        x = GlobalAveragePooling2D()(x)

    Args:
        channels    : Feature map channel depth (must match input).
        r           : Channel reduction ratio (default 8).
        spatial_k   : Spatial attention conv kernel size (default 7).

    Input / Output shapes:
        (B, H, W, C)  →  (B, H, W, C)
    """
    def __init__(self, channels: int, r: int = 8, spatial_k: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.channels  = channels
        self.r         = r
        self.spatial_k = spatial_k
        self.channel   = ChannelAttention(channels, r,        name="ch_attn")
        self.spatial   = SpatialAttention(spatial_k,          name="sp_attn")

    def call(self, x, training=None):
        x = self.channel(x, training=training)
        x = self.spatial(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(channels=self.channels, r=self.r, spatial_k=self.spatial_k)
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np

    B, H, W, C = 2, 6, 6, 64
    dummy = tf.random.normal((B, H, W, C))

    ch  = ChannelAttention(C, r=8)
    sp  = SpatialAttention(kernel_size=7)
    cbam = CBAM(channels=C, r=8, spatial_k=7)

    out_ch   = ch(dummy)
    out_sp   = sp(dummy)
    out_cbam = cbam(dummy)

    assert out_ch.shape   == (B, H, W, C), f"ChannelAttention shape error: {out_ch.shape}"
    assert out_sp.shape   == (B, H, W, C), f"SpatialAttention shape error: {out_sp.shape}"
    assert out_cbam.shape == (B, H, W, C), f"CBAM shape error: {out_cbam.shape}"

    print(f"✅ ChannelAttention : in={dummy.shape} → out={out_ch.shape}")
    print(f"✅ SpatialAttention : in={dummy.shape} → out={out_sp.shape}")
    print(f"✅ CBAM             : in={dummy.shape} → out={out_cbam.shape}")
    print("✅ attention.py OK")
