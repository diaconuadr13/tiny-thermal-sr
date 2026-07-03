"""Keras SR models for full-INT8 TFLite Micro deployment.

Every architecture is built from a restricted op vocabulary with known int8
TFLite Micro kernels: Conv2D, ReLU, tf.nn.depth_to_space (DEPTH_TO_SPACE),
Add. Trained-is-flashed: the same graph is quantized, so no hybrid ops can
appear.

Deviations from the PyTorch thesis models (all documented in the paper):
- ESPCN: Tanh -> ReLU (Tanh int8 LUT exists in TFLM but ReLU is cheaper and
  keeps one activation type across all models).
- FSRCNN: PReLU -> ReLU; ConvTranspose2d(9x9, s2) -> Conv2D(3x3) +
  depth_to_space (sub-pixel head). Referred to as FSRCNN-ds.
- EDSR_Tiny: residual scaling (0.1) is a Rescaling layer during training and
  is folded into the second conv of each block for export (fold_res_scale),
  so the deployed graph is pure Conv2D/Add.

PixelShuffle vs depth_to_space channel order differs in general, but for
1-channel output (grayscale) both reduce to index ry*r+rx — identical.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

DEFAULT_LR_SIZE = (24, 32)  # (height, width) of the LR input


def _depth_to_space(scale: int, name: str):
    return layers.Lambda(
        lambda x: tf.nn.depth_to_space(x, scale), name=name
    )


def build_espcn(scale: int = 2, lr_size=DEFAULT_LR_SIZE,
                widths=(64, 32), first_kernel: int = 5,
                name: str = "espcn") -> keras.Model:
    """ESPCN (Shi 2016), Tanh replaced by ReLU."""
    inp = layers.Input(shape=(*lr_size, 1), name="lr")
    x = layers.Conv2D(widths[0], first_kernel, padding="same",
                      activation="relu", name="conv1")(inp)
    x = layers.Conv2D(widths[1], 3, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.Conv2D(scale ** 2, 3, padding="same", name="conv3")(x)
    out = _depth_to_space(scale, "depth_to_space")(x)
    return keras.Model(inp, out, name=name)


def build_espcn_light(scale: int = 2, lr_size=DEFAULT_LR_SIZE) -> keras.Model:
    """ESPCN with halved widths (32->16), ReLU (as in thesis)."""
    return build_espcn(scale, lr_size, widths=(32, 16), name="espcn_light")


def build_espcn_micro(scale: int = 2, lr_size=DEFAULT_LR_SIZE) -> keras.Model:
    """MCU-class ESPCN: widths 16->8, all 3x3 kernels (as in thesis)."""
    return build_espcn(scale, lr_size, widths=(16, 8), first_kernel=3,
                       name="espcn_micro")


def build_fsrcnn(scale: int = 2, lr_size=DEFAULT_LR_SIZE,
                 d: int = 56, s: int = 12, m: int = 4) -> keras.Model:
    """FSRCNN-ds: Dong 2016 with ReLU and a sub-pixel head instead of the
    9x9 stride-2 ConvTranspose (no fully-supported int8 TFLM kernel)."""
    inp = layers.Input(shape=(*lr_size, 1), name="lr")
    x = layers.Conv2D(d, 5, padding="same", activation="relu",
                      name="feature_extraction")(inp)
    x = layers.Conv2D(s, 1, padding="same", activation="relu",
                      name="shrinking")(x)
    for i in range(m):
        x = layers.Conv2D(s, 3, padding="same", activation="relu",
                          name=f"mapping_{i}")(x)
    x = layers.Conv2D(d, 1, padding="same", activation="relu",
                      name="expanding")(x)
    x = layers.Conv2D(scale ** 2, 3, padding="same", name="subpixel_conv")(x)
    out = _depth_to_space(scale, "depth_to_space")(x)
    return keras.Model(inp, out, name="fsrcnn_ds")


def build_edsr_tiny(scale: int = 2, lr_size=DEFAULT_LR_SIZE,
                    num_feats: int = 32, num_blocks: int = 8,
                    res_scale: float = 0.1,
                    fold_res_scale: bool = False) -> keras.Model:
    """EDSR-Tiny: head -> N x ResBlock -> conv -> global residual -> subpixel.

    fold_res_scale=False: training graph, res_scale applied via Rescaling.
    fold_res_scale=True: deployment graph without Rescaling; use
    fold_edsr_res_scale() to transfer trained weights into it.
    """
    inp = layers.Input(shape=(*lr_size, 1), name="lr")
    head = layers.Conv2D(num_feats, 3, padding="same", name="head")(inp)
    x = head
    for i in range(num_blocks):
        y = layers.Conv2D(num_feats, 3, padding="same", activation="relu",
                          name=f"block{i}_conv1")(x)
        y = layers.Conv2D(num_feats, 3, padding="same",
                          name=f"block{i}_conv2")(y)
        if not fold_res_scale:
            y = layers.Rescaling(res_scale, name=f"block{i}_res_scale")(y)
        x = layers.Add(name=f"block{i}_add")([x, y])
    x = layers.Conv2D(num_feats, 3, padding="same", name="body_end")(x)
    x = layers.Add(name="global_add")([head, x])
    x = layers.Conv2D(scale ** 2, 3, padding="same", name="subpixel_conv")(x)
    out = _depth_to_space(scale, "depth_to_space")(x)
    return keras.Model(inp, out, name="edsr_tiny")


def fold_edsr_res_scale(trained: keras.Model, scale: int = 2,
                        lr_size=DEFAULT_LR_SIZE,
                        res_scale: float = 0.1) -> keras.Model:
    """Return an equivalent EDSR-Tiny graph with res_scale baked into each
    block's second conv (kernel and bias scaled), removing the Rescaling
    ops so the int8 graph is pure Conv2D/Add."""
    num_blocks = sum(1 for l in trained.layers if l.name.endswith("_add")
                     and l.name != "global_add")
    folded = build_edsr_tiny(scale, lr_size, num_blocks=num_blocks,
                             res_scale=res_scale, fold_res_scale=True)
    for layer in folded.layers:
        if not layer.weights:
            continue
        src = trained.get_layer(layer.name)
        weights = src.get_weights()
        if layer.name.endswith("_conv2"):
            weights = [w * res_scale for w in weights]
        layer.set_weights(weights)
    return folded


MODEL_BUILDERS = {
    "espcn": build_espcn,
    "espcn_light": build_espcn_light,
    "espcn_micro": build_espcn_micro,
    "fsrcnn_ds": build_fsrcnn,
    "edsr_tiny": build_edsr_tiny,
}


def get_model(name: str, scale: int = 2, lr_size=DEFAULT_LR_SIZE) -> keras.Model:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"unknown model '{name}', have {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](scale=scale, lr_size=lr_size)
