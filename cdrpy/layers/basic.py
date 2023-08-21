"""

"""

from __future__ import annotations

import functools

import typing as t
import tensorflow as tf

from tensorflow import keras
from keras import layers


class MLPBlock(layers.Layer):
    """"""

    def __init__(
        self,
        units: int,
        activation: t.Any,
        use_batch_norm: bool = False,
        use_dropout: bool = False,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super(MLPBlock, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self._create_layers()

    def call(self, inputs: t.Any, training: bool = True, **kwargs) -> t.Any:
        x = self.dense(inputs)
        if training and self.batch_norm is not None:
            x = self.batch_norm(x)
        if training and self.dropout is not None:
            x = self.dropout(x)
        return x

    def _create_layers(self) -> None:
        """Creates the layers for the MLP block."""
        self.dense = layers.Dense(self.units, activation=self.activation)
        self.batch_norm = None
        self.dropout = None
        if self.use_batch_norm:
            self.batch_norm = layers.BatchNormalization()
        if self.use_dropout:
            self.dropout = layers.Dropout(self.dropout_rate)
