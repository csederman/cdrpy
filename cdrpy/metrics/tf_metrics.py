"""
Tensorflow metrics
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K

from tensorflow import keras


@keras.utils.register_keras_serializable(package="cdrpy")
def r2(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Calculates r-squared."""
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (ss_res / ss_tot)


@keras.utils.register_keras_serializable(package="cdrpy")
def pearson(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Calculates pearson correlation."""
    return tfp.stats.correlation(y_true, y_pred)


@keras.utils.register_keras_serializable(package="cdrpy")
def rmse(s_true, s_pred):
    """Calculates root mean squared error."""
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))
