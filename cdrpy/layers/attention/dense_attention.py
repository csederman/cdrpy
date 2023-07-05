"""

"""

from __future__ import annotations

import typing as t
import tensorflow as tf

from tensorflow import keras


class DenseAttention(keras.layers.Layer):
    """Attention mechanism layer for dense inputs.

    The original implementation of this layer can be found at:
        https://github.com/prassepaul/mlmed_transfer_learning

    Args:
        units: The number of units for the dense layer.
    """

    def __init__(self, units: int):
        super(DenseAttention, self).__init__()
        self.dense = keras.layers.Dense(
            units, activation=keras.activations.softmax
        )

    def call(self, inputs):
        alphas = self.dense(inputs)
        return tf.multiply(inputs, alphas)
