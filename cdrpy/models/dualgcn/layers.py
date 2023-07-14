"""

"""

from __future__ import annotations

import typing as t
import tensorflow as tf

keras = tf.keras  # pylance issue #1066

import keras.backend as K


# class DualGCNGraphLayer(keras.layers.Layer):
#     """Reimplementation of DualGCN GraphLayer."""

#     def __init__(
#         self,
#         step_num: int = 1,
#         activation: str | t.Callable | None = None,
#         **kwargs,
#     ) -> None:
#         self.step_num = step_num
#         self.activation = keras.activations.get(activation)
#         self.supports_masking = True
#         super(DualGCNGraphLayer, self).__init__(**kwargs)

#     def get_config(self):
#         config = {
#             "step_num": self.step_num,
#             "activation": self.activation,
#         }
#         base_config = super(DualGCNGraphLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     def _get_walked_edges(self, edges, step_num):
#         if step_num <= 1:
#             return edges
#         deeper = self._get_walked_edges(
#             K.batch_dot(edges, edges), step_num // 2
#         )
#         if step_num % 2 == 1:
#             deeper += edges
#         return K.cast(K.greater(deeper, 0.0), K.floatx())

#     def call(self, inputs, **kwargs):
#         features, edges = inputs
#         edges = K.cast(edges, K.floatx())
#         if self.step_num > 1:
#             edges = self._get_walked_edges(edges, self.step_num)
#         outputs = self.activation(self._call(features, edges))
#         return outputs

#     def _call(self, features, edges):
#         raise NotImplementedError(
#             "The class is not intended to be used directly."
#         )


# class DualGCNGraphConv(DualGCNGraphLayer):
#     def __init__(
#         self,
#         units,
#         kernel_initializer="glorot_uniform",
#         kernel_regularizer=None,
#         kernel_constraint=None,
#         use_bias=True,
#         bias_initializer="zeros",
#         bias_regularizer=None,
#         bias_constraint=None,
#         **kwargs,
#     ):
#         self.units = units
#         self.kernel_initializer = keras.initializers.get(kernel_initializer)
#         self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
#         self.kernel_constraint = keras.constraints.get(kernel_constraint)
#         self.use_bias = use_bias
#         self.bias_initializer = keras.initializers.get(bias_initializer)
#         self.bias_regularizer = keras.regularizers.get(bias_regularizer)
#         self.bias_constraint = keras.constraints.get(bias_constraint)

#         self.W, self.b = None, None
#         super(DualGCNGraphConv, self).__init__(**kwargs)

#     def get_config(self):
#         config = {
#             "units": self.units,
#             "kernel_initializer": keras.initializers.serialize(
#                 self.kernel_initializer
#             ),
#             "kernel_regularizer": keras.regularizers.serialize(
#                 self.kernel_regularizer
#             ),
#             "kernel_constraint": keras.constraints.serialize(
#                 self.kernel_constraint
#             ),
#             "use_bias": self.use_bias,
#             "bias_initializer": keras.initializers.serialize(
#                 self.bias_initializer
#             ),
#             "bias_regularizer": keras.regularizers.serialize(
#                 self.bias_regularizer
#             ),
#             "bias_constraint": keras.constraints.serialize(
#                 self.bias_constraint
#             ),
#         }
#         base_config = super(DualGCNGraphConv, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     def build(self, input_shape):
#         feature_dim = input_shape[0][2]
#         self.W = self.add_weight(
#             shape=(feature_dim, self.units),
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
#             name="{}_W".format(self.name),
#         )
#         if self.use_bias:
#             self.b = self.add_weight(
#                 shape=(self.units,),
#                 initializer=self.bias_initializer,
#                 regularizer=self.bias_regularizer,
#                 constraint=self.bias_constraint,
#                 name="{}_b".format(self.name),
#             )
#         super(DualGCNGraphConv, self).build(input_shape)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0][:2] + (self.units,)

#     def compute_mask(self, inputs, mask=None):
#         if mask is None:
#             return None
#         return mask[0]

#     def _call(self, features, edges):
#         features = K.dot(features, self.W)
#         if self.use_bias:
#             features += self.b
#         if self.step_num > 1:
#             edges = self._get_walked_edges(edges, self.step_num)
#         return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features)


# class DualGCNGraphPool(DualGCNGraphLayer):
#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def compute_mask(self, inputs, mask=None):
#         if mask is None:
#             return None
#         return mask[0]


# class DualGCNGraphMaxPool(DualGCNGraphPool):
#     NEG_INF = -1e38

#     def _call(self, features, edges):
#         node_num = K.shape(features)[1]
#         features = K.tile(
#             K.expand_dims(features, axis=1), K.stack([1, node_num, 1, 1])
#         ) + K.expand_dims((1.0 - edges) * self.NEG_INF, axis=-1)
#         return K.max(features, axis=2)


# class DualGCNGraphAveragePool(DualGCNGraphPool):
#     def _call(self, features, edges):
#         return K.batch_dot(
#             K.permute_dimensions(edges, (0, 2, 1)), features
#         ) / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())
