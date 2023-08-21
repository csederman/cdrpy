"""
Main training loop for ScreenDL.

FIXME: this training loop is not model-specific: move this to models/train.py
"""

from __future__ import annotations

import os
import typing as t

from tensorflow import keras

from cdrpy.metrics import tf_metrics

if t.TYPE_CHECKING:
    from cdrpy.types import PathLike
    from cdrpy.data.datasets import Dataset


def train(
    model: keras.Model,
    opt: keras.optimizers.Optimizer,
    train_ds: Dataset,
    val_ds: Dataset | None = None,
    batch_size: int = 32,
    epochs: int = 10,
    save_dir: PathLike | None = None,
    log_dir: PathLike | None = None,
    early_stopping: bool = False,
    tensorboard: bool = False,
) -> keras.Model:
    """Train the ScreenDL model.

    Parameters
    ----------
        model:
        opt:
        train_ds:
        val_ds:
        epochs:
        batch_size:
        save_dir:
        log_dir:
        early_stopping:
        tensorboard:

    Returns
    -------
        The trained `keras.Model` instance.
    """
    # TODO: add early stopping and tensorboard callbacks

    callbacks = []

    if early_stopping:
        # FIXME: do we want to restore best weights?
        callbacks.append(
            keras.callbacks.EarlyStopping(
                "val_loss",
                patience=5,
                restore_best_weights=True,
                start_from_epoch=5,
            )
        )
    if tensorboard:
        if log_dir is None:
            raise ValueError(
                "log_dir must be specified when using tensorboard"
            )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        # FIXME: check what kind of paths this works with
        callbacks.append(
            keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        )

    train_tfds = (
        train_ds.encode_tf()
        .shuffle(10000, reshuffle_each_iteration=True)
        .batch(batch_size)
    )
    val_tfds = (
        val_ds.encode_tf()
        .shuffle(10000, reshuffle_each_iteration=True)
        .batch(batch_size)
    )

    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=["mse", tf_metrics.pearson],
    )

    hx = model.fit(
        train_tfds,
        epochs=epochs,
        validation_data=val_tfds,
        callbacks=callbacks,
    )

    if save_dir is not None:
        model.save(os.path.join(save_dir, "model"))
        model.save_weights(os.path.join(save_dir, "weights"))

    return model
