"""Train the Keras SR models on the FLIR-derived thermal pairs.

Usage:
    python train_tf.py --config configs/espcn_micro.yaml
    python train_tf.py --model espcn_micro          # config-free default

Outputs per run (runs/<model>/):
    best.weights.h5, last.weights.h5, training_log.csv, summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
import yaml

import models_tf

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "flir_thermal_x2"

DEFAULTS = {
    "model": None,
    "scale": 2,
    "lr_size": [24, 32],  # (height, width)
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "seed": 42,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def _load_pair(lr_path: tf.Tensor, hr_path: tf.Tensor):
    lr = tf.io.decode_png(tf.io.read_file(lr_path), channels=1)
    hr = tf.io.decode_png(tf.io.read_file(hr_path), channels=1)
    return (tf.cast(lr, tf.float32) / 255.0,
            tf.cast(hr, tf.float32) / 255.0)


def _augment(lr: tf.Tensor, hr: tf.Tensor):
    # Horizontal/vertical flips only: 90-degree rotations would swap the
    # non-square 32x24 spatial dims and break batching.
    if tf.random.uniform(()) > 0.5:
        lr, hr = tf.image.flip_left_right(lr), tf.image.flip_left_right(hr)
    if tf.random.uniform(()) > 0.5:
        lr, hr = tf.image.flip_up_down(lr), tf.image.flip_up_down(hr)
    return lr, hr


def make_dataset(split: str, batch_size: int, augment: bool,
                 shuffle: bool) -> tf.data.Dataset:
    lr_dir = DATA_ROOT / split / "LR"
    hr_dir = DATA_ROOT / split / "HR"
    names = sorted(p.name for p in lr_dir.glob("*.png"))
    if not names:
        raise RuntimeError(f"no pairs in {lr_dir}; run data_prep first")
    lr_paths = [str(lr_dir / n) for n in names]
    hr_paths = [str(hr_dir / n) for n in names]
    ds = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
    ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(len(names), reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def bicubic_baseline(scale: int) -> dict:
    """PSNR/SSIM of bicubic upscaling on the val split (reference floor)."""
    ds = make_dataset("val", batch_size=32, augment=False, shuffle=False)
    p_vals, s_vals = [], []
    for lr, hr in ds:
        up = tf.image.resize(lr, hr.shape[1:3], method="bicubic")
        up = tf.clip_by_value(up, 0.0, 1.0)
        p_vals.append(tf.image.psnr(hr, up, max_val=1.0))
        s_vals.append(tf.image.ssim(hr, up, max_val=1.0))
    return {
        "psnr": float(tf.reduce_mean(tf.concat(p_vals, 0))),
        "ssim": float(tf.reduce_mean(tf.concat(s_vals, 0))),
    }


def train(config: dict) -> dict:
    set_seed(config["seed"])
    name = config["model"]
    lr_size = tuple(config["lr_size"])
    run_dir = REPO_ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = models_tf.get_model(name, scale=config["scale"], lr_size=lr_size)
    model.compile(
        optimizer=keras.optimizers.Adam(config["learning_rate"]),
        loss="mae",
        metrics=[psnr, ssim],
    )

    train_ds = make_dataset("train", config["batch_size"],
                            augment=True, shuffle=True)
    val_ds = make_dataset("val", config["batch_size"],
                          augment=False, shuffle=False)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best.weights.h5"), monitor="val_psnr",
            mode="max", save_best_only=True, save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_psnr", mode="max", factor=0.5,
            patience=20, min_lr=1e-5),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=config["epochs"], callbacks=callbacks,
                        verbose=2)

    model.save_weights(run_dir / "last.weights.h5")
    best_epoch = int(np.argmax(history.history["val_psnr"]))
    summary = {
        "model": name,
        "params": int(model.count_params()),
        "config": config,
        "best_epoch": best_epoch,
        "best_val_psnr": float(history.history["val_psnr"][best_epoch]),
        "best_val_ssim": float(history.history["val_ssim"][best_epoch]),
        "bicubic_baseline": bicubic_baseline(config["scale"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML config")
    parser.add_argument("--model", help="model name (overrides config)")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = dict(DEFAULTS)
    if args.config:
        config.update(yaml.safe_load(args.config.read_text()))
    if args.model:
        config["model"] = args.model
    if args.epochs:
        config["epochs"] = args.epochs
    if not config["model"]:
        parser.error("need --model or a config with 'model'")
    train(config)


if __name__ == "__main__":
    main()
