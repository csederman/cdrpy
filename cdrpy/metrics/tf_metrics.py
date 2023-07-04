"""
Tensorflow metrics
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

keras = tf.keras

from keras.dtensor import utils as dtensor_utils


def r2(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Calculates r squared."""
    ss_res = keras.backend.sum(keras.backend.square(y_true - y_pred))
    ss_tot = keras.backend.sum(
        keras.backend.square(y_true - keras.backend.mean(y_true))
    )
    return 1 - (ss_res / ss_tot)


def pearson(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Calculates pearson correlation."""
    return tfp.stats.correlation(y_true, y_pred)


class MeanPearsonCorrelation(keras.metrics.MeanMetricWrapper):
    """Mean pearson correlation over batches."""

    @dtensor_utils.inject_mesh
    def __init__(self, name="pearson", dtype=None):
        super().__init__(pearson, name, dtype=dtype)
