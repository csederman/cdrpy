"""
Tensorflow utilities.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

keras = tf.keras


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
