"""

"""

from __future__ import annotations

import typing as t
import tensorflow as tf
import keras.backend as K

from tensorflow import keras
from keras import layers

from cdrpy.layers.graph import GraphConvBlock


def _create_mut_subnetwork(mut_dim: int) -> keras.Sequential:
    """"""
    model = keras.Sequential(
        [
            layers.Input((1, mut_dim, 1), name="cell_mut_input"),
            layers.Conv2D(
                50,
                activation="tanh",
                kernel_size=(1, 700),
                strides=(1, 5),
                name="cell_mut_conv2d_1",
            ),
            layers.MaxPool2D(pool_size=(1, 5), name="cell_mut_mpool2d_1"),
            layers.Conv2D(
                30,
                activation="relu",
                kernel_size=(1, 5),
                strides=(1, 2),
                name="cell_mut_conv2d_2",
            ),
            layers.MaxPooling2D(pool_size=(1, 10), name="cell_mut_mpool2d_2"),
            layers.Flatten(name="cell_mut_flatten_1"),
            layers.Dense(100, activation="relu", name="cell_mut_dense_1"),
            layers.Dropout(0.1, name="cell_mut_dropout_1"),
        ],
        name="cell_mut_subnet",
    )

    return model


def _create_exp_subnetwork(
    exp_dim: int, exp_norm: layers.Normalization | None = None
) -> keras.Model:
    """Creates the gene expression subnetwork."""
    exp_input = layers.Input((exp_dim,), name="cell_exp_input")
    if exp_norm is not None:
        if not exp_norm.is_adapted:
            # FIXME: change this to a warning since you can still adapt later
            raise ValueError("requires adapted normalization layer...")
        exp_input = exp_norm(exp_input)
    x = layers.Dense(256, activation="tanh", name="cell_exp_dense_1")(
        exp_input
    )
    x = layers.BatchNormalization(name="cell_exp_bnorm_1")(x)
    x = layers.Dropout(0.1, name="cell_exp_dropout_1")(x)
    x = layers.Dense(100, activation="relu", name="cell_exp_dense_2")(x)

    model = keras.Model(inputs=exp_input, outputs=x, name="cell_exp_subnet")

    return model


def _create_methyl_subnetwork(methyl_dim: int) -> keras.Sequential:
    """Creates the methylation subnetwork."""
    # FIXME: add option for normalization layer since this is float data
    model = keras.Sequential(
        [
            layers.Input((methyl_dim,), name="cell_methyl_input"),
            layers.Dense(256, activation="tanh", name="cell_methyl_dense_1"),
            layers.BatchNormalization(name="cell_methyl_bnorm_1"),
            layers.Dropout(0.1, name="cell_methyl_dropout_1"),
            layers.Dense(100, activation="relu", name="cell_methyl_dense_2"),
        ],
        name="cell_methyl_subnet",
    )

    return model


def _create_cell_subnetwork(
    exp_dim: int,
    mut_dim: int,
    methyl_dim: int | None = None,
    exp_norm: layers.Normalization | None = None,
) -> keras.Model:
    """"""
    exp_subnet = _create_exp_subnetwork(exp_dim, exp_norm)
    mut_subnet = _create_mut_subnetwork(mut_dim)
    subnet_inputs = [exp_subnet.input, mut_subnet.input]
    subnet_outputs = [exp_subnet.output, mut_subnet.output]
    if methyl_dim is not None:
        methyl_subnet = _create_methyl_subnetwork(methyl_dim)
        subnet_inputs.append(methyl_subnet.input)
        subnet_outputs.append(methyl_subnet.output)
    output = layers.Concatenate(name="cell_subnet_concat")(subnet_outputs)

    return keras.Model(
        inputs=subnet_inputs, outputs=output, name="cell_subnet"
    )


def _create_drug_subnetwork(feat_dim: int) -> keras.Model:
    """Creates the drug subnetwork."""
    feat_input = layers.Input((None, feat_dim), name="drug_feat_input")
    adj_input = layers.Input((None, None), name="drug_adj_input")

    x = [feat_input, adj_input]
    x = GraphConvBlock(units=256, step_num=1, name="drug_gconv_1")(x)
    x = [x, adj_input]
    x = GraphConvBlock(units=256, step_num=1, name="drug_gconv_2")(x)
    x = [x, adj_input]
    x = GraphConvBlock(units=256, step_num=1, name="drug_gconv_3")(x)
    x = [x, adj_input]
    x = GraphConvBlock(units=100, step_num=1, name="drug_gconv_4")(x)
    x = layers.GlobalMaxPooling1D(name="drug_pool")(x)

    return keras.Model(
        inputs=[feat_input, adj_input], outputs=x, name="drug_subnet"
    )


def create_model(
    cell_exp_dim: int,
    cell_mut_dim: int,
    drug_feat_dim: int,
    cell_methyl_dim: int | None = None,
    cell_exp_norm: layers.Normalization | None = None,
) -> keras.Model:
    """Creates the DeepCDR model."""
    cell_subnet = _create_cell_subnetwork(
        cell_exp_dim, cell_mut_dim, cell_methyl_dim, cell_exp_norm
    )
    drug_subnet = _create_drug_subnetwork(drug_feat_dim)

    subnet_inputs = [*cell_subnet.input, *drug_subnet.input]
    subnet_outputs = [cell_subnet.output, drug_subnet.output]

    x = layers.Concatenate(name="shared_concat_1")(subnet_outputs)
    x = layers.Dense(300, activation="tanh", name="shared_dense_1")(x)
    x = layers.Dropout(0.1, name="shared_dropout_1")(x)
    x = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = layers.Lambda(lambda x: K.expand_dims(x, axis=1))(x)
    x = layers.Conv2D(
        30,
        kernel_size=(1, 150),
        strides=(1, 1),
        activation="relu",
        name="shared_conv2d_1",
    )(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), name="shared_mpool2d_1")(x)
    x = layers.Conv2D(
        filters=10,
        kernel_size=(1, 5),
        strides=(1, 1),
        activation="relu",
        name="shared_conv2d_2",
    )(x)
    x = layers.MaxPooling2D(pool_size=(1, 3), name="shared_mpool2d_2")(x)
    x = layers.Conv2D(
        filters=5,
        kernel_size=(1, 5),
        strides=(1, 1),
        activation="relu",
        name="shared_conv2d_3",
    )(x)
    x = layers.MaxPooling2D(pool_size=(1, 3), name="shared_mpool2d_3")(x)
    x = layers.Dropout(0.1, name="shared_dropout_2")(x)
    x = layers.Flatten(name="shared_flat_1")(x)
    x = layers.Dropout(0.2, name="shared_dropout_3")(x)
    output = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=subnet_inputs, outputs=output, name="DeepCDR")

    return model
