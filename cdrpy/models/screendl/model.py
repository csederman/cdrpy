"""

"""

from __future__ import annotations

import typing as t
import tensorflow as tf

keras = tf.keras  # pylance issue #1066

from keras import Model
from keras import layers, losses, optimizers

from cdrpy.data._datasets import Dataset
from cdrpy.data.types import EncodedDataset, EncodedFeatures


def _create_cell_subnetwork(
    exp_dim: int, exp_norm: layers.Normalization | None = None
) -> keras.Model:
    """"""
    exp_input = layers.Input((exp_dim,), name="exp_input")
    if exp_norm is not None:
        if not exp_norm.is_adapted:
            # FIXME: change this to a warning since you can still adapt later
            raise ValueError("requires adapted normalization layer...")
        exp_input = exp_norm(exp_input)
    x = layers.GaussianNoise(0.1, name="exp_noise_1")(exp_input)
    x = layers.Dense(exp_dim, "relu", name="exp_dense_1")(x)
    x = layers.BatchNormalization(name="exp_bnorm_1")(x)
    x = layers.Dense(int(exp_dim / 2), "relu", name="exp_dense_2")(x)
    x = layers.BatchNormalization(name="exp_bnorm_2")(x)
    x = layers.Dense(int(exp_dim / 4), "relu", name="exp_dense_3")(x)
    x = layers.BatchNormalization(name="exp_bnorm_3")(x)
    x = layers.Dense(int(exp_dim / 8), "relu", name="exp_dense_4")(x)
    x = layers.BatchNormalization(name="exp_bnorm_4")(x)

    model = keras.Model(inputs=exp_input, outputs=x, name="cell_subnet")

    return model


# def _build_dna_model(dim: int) -> keras.Sequential:
#     """"""
#     model = keras.Sequential(
#         [
#             layers.Input((dim,), name="dna_input"),
#             layers.Dense(dim, "relu", name="dna_dense_1"),
#             layers.BatchNormalization(name="dna_bnorm_1"),
#             layers.Dense(int(dim / 2), "relu", name="dna_dense_2"),
#             layers.BatchNormalization(name="dna_bnorm_2"),
#             layers.Dense(int(dim / 4), "relu", name="dna_dense_3"),
#             layers.BatchNormalization(name="dna_bnorm_3"),
#             layers.Dense(int(dim / 8), "relu", name="dna_dense_4"),
#             layers.BatchNormalization(name="dna_bnorm_4"),
#         ]
#     )

#     return model


def _create_drug_subnetwork(drug_dim: int) -> keras.Sequential:
    """"""

    # NOTE: Ideker DrugCell dims (except for first dense layer I think)
    model = keras.Sequential(
        [
            layers.Input((drug_dim,), name="fp_input"),
            layers.Dense(drug_dim, "relu", name="fp_dense_1"),
            layers.BatchNormalization(name="fp_bnorm_1"),
            # layers.Dense(100, "relu", name="fp_dense2"),
            # layers.Dense(50, "relu", name="fp_dense3"),
            # layers.Dense(10, "relu", name="fp_dense4"),
            layers.Dense(int(drug_dim / 2), "relu", name="fp_dense_2"),
            layers.BatchNormalization(name="fp_bnorm_2"),
            layers.Dense(int(drug_dim / 4), "relu", name="fp_dense_3"),
            layers.BatchNormalization(name="fp_bnorm_3"),
            layers.Dense(int(drug_dim / 8), "relu", name="fp_dense_4"),
            layers.BatchNormalization(name="fp_bnorm_4"),
        ],
        name="drug_subnet",
    )

    return model


def create_model(
    cell_dim: int,
    drug_dim: int,
    cell_exp_norm: layers.Normalization | None = None,
) -> keras.Model:
    """"""
    cell_subnet = _create_cell_subnetwork(cell_dim, cell_exp_norm)
    drug_subnet = _create_drug_subnetwork(drug_dim)
    subnet_inputs = [cell_subnet.input, drug_subnet.input]
    subnet_outputs = [cell_subnet.output, drug_subnet.output]

    dim = sum((cell_subnet.output_shape[1], drug_subnet.output_shape[1]))

    x = layers.Concatenate(name="concat")(subnet_outputs)
    x = layers.Dense(dim, "relu", name="final_dense1")(x)
    x = layers.BatchNormalization(name="final_bnorm_1")(x)
    x = layers.Dense(int(dim / 2), "relu", name="final_dense_2")(x)
    x = layers.BatchNormalization(name="final_bnorm_2")(x)
    x = layers.Dense(int(dim / 4), "relu", name="final_dense_3")(x)
    x = layers.BatchNormalization(name="final_bnorm_3")(x)
    x = layers.Dense(int(dim / 8), "relu", name="final_dense_4")(x)
    x = layers.BatchNormalization(name="final_bnorm_4")(x)
    x = layers.Dense(int(dim / 10), "relu", name="final_dense_5")(x)
    x = layers.BatchNormalization(name="final_bnorm_5")(x)
    output = layers.Dense(1, "linear", name="final_activation")(x)

    return Model(inputs=subnet_inputs, outputs=output, name="ScreenDL")


# class ScreenDL:
#     """"""

#     _channel_builders: dict[str, t.Callable[[int], keras.Sequential]] = {
#         "dna": _build_dna_model,
#         "rna": _build_rna_model,
#         "fp": _build_fp_model,
#     }

#     def __init__(
#         self,
#         channel_names: list[str],
#         channel_dims: list[int],
#         opt: keras.optimizers.Optimizer,
#         loss_fn: keras.losses.Loss,
#         metrics: t.Iterable[keras.metrics.Metric],
#     ) -> None:
#         assert len(channel_names) == len(channel_dims)
#         self._channel_names = channel_names
#         self._channel_dims = channel_dims
#         self._opt = opt
#         self._loss_fn = loss_fn
#         self._metrics = metrics
#         self._init_model()

#     @property
#     def model(self) -> keras.Model:
#         return self._model

#     @property
#     def opt(self) -> keras.optimizers.Optimizer:
#         return self._opt

#     @property
#     def loss_fn(self) -> keras.losses.Loss:
#         return self._loss_fn

#     @property
#     def metrics(self) -> t.Iterable[keras.metrics.Metric]:
#         return self._metrics

#     def _init_model(self) -> None:
#         """Initialize the keras model."""
#         channels = []
#         for channel_name, dim in zip(self._channel_names, self._channel_dims):
#             builder = self._channel_builders[channel_name]
#             channel = builder(dim)
#             channels.append(channel)

#         self._model = _build_model(channels)
#         # self._model.compile(optimizer=self.opt, loss=self.loss_fn, )

#     def train_step(self, ds: Dataset, batch_size: int = 32) -> None:
#         """Perform a single training iteration/epoch."""
#         batch_generator = ds.encode_batches(["rna"], ["fp"], batch_size)
#         for step, batch in enumerate(batch_generator):
#             X, y, *_ = batch
#             y = y.reshape(-1, 1)

#             with tf.GradientTape() as tape:
#                 preds = self.model(X, training=True)
#                 loss_value = self.loss_fn(y, preds)

#             grads = tape.gradient(loss_value, self.model.trainable_weights)
#             self.opt.apply_gradients(zip(grads, self.model.trainable_weights))

#             for metric in self.metrics:
#                 metric.update_state(y, preds)

#         for metric in self.metrics:
#             print(step, metric.name, float(metric.result()))
#             # if step % 100 == 0:
#             #     for metric_fn in self.metrics:
#             #         batch_metric = metric_fn(y, preds)
