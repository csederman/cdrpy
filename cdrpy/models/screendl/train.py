"""
Main training loop for ScreenDL.

FIXME: this training loop is not model-specific: move this to models/train.py
"""

from __future__ import annotations

import os
import numpy as np
import typing as t

from scipy import stats
from tensorflow import keras

from cdrpy.metrics import tf_metrics
from cdrpy.mapper import BatchedResponseGenerator

if t.TYPE_CHECKING:
    from cdrpy.types import PathLike
    from cdrpy.data.datasets import Dataset


def make_weights(ds: Dataset) -> np.ndarray:
    """Create continuous sample weights."""
    grouped = ds.obs.groupby("drug_id")
    sample_weights = []
    for _, group in grouped:
        y = group["label"]
        y_std = (y - y.min()) / (y.max() - y.min())
        y_scaled = y_std * (1 - (-1)) + (-1)
        sample_weights.extend(list(1 + abs(y_scaled)))

    return np.asanyarray(sample_weights)


def make_dense_weights(
    ds: Dataset, alpha: float = 0.5, epsilon: float = 1e-4
) -> np.ndarray:
    """Create continuous sample weights."""
    grouped = ds.obs.groupby("drug_id")
    sample_weights = []
    for _, group in grouped:
        Y = group["label"]
        kernel = stats.gaussian_kde(Y)

        Z = kernel(Y)
        Z_std = (Z - Z.min()) / (Z.max() - Z.min())

        weights = np.clip((1 - (alpha * Z_std)), a_min=epsilon, a_max=None)
        scaled_weights = weights / weights.mean()

        sample_weights.extend(list(scaled_weights))

    return np.asanyarray(sample_weights)


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
                patience=10,
                restore_best_weights=True,
                start_from_epoch=5,
                verbose=1,
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

    train_gen = BatchedResponseGenerator(train_ds, batch_size)
    val_gen = BatchedResponseGenerator(val_ds, batch_size)

    # FIXME: should just pass in the generator objects since I don't need
    #   the datasets

    # FIXME: add optional param for use of sample_weights
    # FIXME: add parameter for weight alpha as tunable hparam
    # FIXME: move make_dense_weights to core.utils.sample_weights

    train_seq = train_gen.flow(
        train_ds.cell_ids,
        train_ds.drug_ids,
        targets=train_ds.labels,
        # sample_weights=make_dense_weights(train_ds, alpha=1),
        shuffle=True,
        seed=4114,
    )

    val_seq = val_gen.flow(
        val_ds.cell_ids,
        val_ds.drug_ids,
        targets=val_ds.labels,
        # sample_weights=make_dense_weights(val_ds, alpha=1),
        shuffle=False,
    )

    # train_tfds = train_ds.encode_tf().shuffle(10000).batch(batch_size)
    # val_tfds = val_ds.encode_tf().batch(batch_size)

    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=[tf_metrics.pearson, tf_metrics.rmse],
    )

    hx = model.fit(
        # train_tfds,
        train_seq,
        epochs=epochs,
        # validation_data=val_tfds,
        validation_data=val_seq,
        callbacks=callbacks,
    )

    if save_dir is not None:
        model.save(os.path.join(save_dir, "model"))
        model.save_weights(os.path.join(save_dir, "weights"))

    return model
