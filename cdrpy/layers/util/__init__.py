"""
Suite of simple utility layers for cancer drug response prediction.
"""

from __future__ import annotations

import typing as t
import tensorflow as tf

from tensorflow import keras


class TensorDot(keras.layers.Layer):
    """Tensor contraction of a and b along specified axes and outer product.

    The original implementation of this layer can be found at:
        https://github.com/prassepaul/mlmed_transfer_learning

    Args:
        axes: Either a scalar N, or a list or an int32 Tensor of shape [2, k].
    """

    def __init__(self, axes: int = -1):
        super(TensorDot, self).__init__()
        self.axes = axes

    def call(self, inputs):
        return tf.tensordot(inputs[0], inputs[1], axes=self.axes)


class ReduceSum(keras.layers.Layer):
    """Computes the sum of elements across dimensions of a tensor.

    The original implementation of this layer can be found at:
        https://github.com/prassepaul/mlmed_transfer_learning
    """

    def __init__(self, axis: t.Any | None):
        super(ReduceSum, self).__init__()
        self.axis = axis

    def call(self, input):
        return tf.reduce_sum(input, axis=self.axis)


class ExpandDim(keras.layers.Layer):
    """Adds a 1-sized dimension at index "axis".

    The original implementation of this layer can be found at:
        https://github.com/prassepaul/mlmed_transfer_learning
    """

    def __init__(self, axis: int = -1):
        super(ExpandDim, self).__init__()
        self.axis = axis

    def call(self, input: t.Any):
        return keras.backend.expand_dims(input, axis=self.axis)


class Squeeze(keras.layers.Layer):
    """Removes a 1-dimension from the tensor at index "axis".

    The original implementation of this layer can be found at:
        https://github.com/prassepaul/mlmed_transfer_learning
    """

    def __init__(self, axis):
        super(Squeeze, self).__init__()
        self.axis = axis

    def call(self, input):
        return keras.backend.squeeze(input, axis=self.axis)


class Embedding(keras.layers.Layer):
    """Simple embedding layer implementation.

    The original implementation of this layer can be found at:
        https://github.com/prassepaul/mlmed_transfer_learning
    """

    def __init__(self, x, y):
        super(Embedding, self).__init__()
        self.embedding_matrix = tf.Variable(tf.random.normal((x, y)))

    def call(self, input):
        return tf.nn.embedding_lookup(
            self.embedding_matrix, tf.cast(input, dtype=tf.int32)
        )
