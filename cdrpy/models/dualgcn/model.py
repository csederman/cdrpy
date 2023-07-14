"""

"""

from __future__ import annotations

import typing as t
import tensorflow as tf

from tensorflow import keras
from keras import layers

from cdrpy.layers.graph import GraphConvBlock


def _create_cell_channel(
    cell_dim: int, cell_norm: layers.Normalization | None = None
) -> keras.Model:
    """"""
    cell_feat_input = keras.Input(
        shape=(None, cell_dim), name="cell_feat_input"
    )
    cell_adj_input = keras.Input(shape=(None, None), name="cell_adj_input")

    if cell_norm is not None:
        cell_feat_input = cell_norm(cell_feat_input)
    x = layers.Dense(32, activation="tanh", name="cell_dense1")(
        cell_feat_input
    )
    x = layers.Dropout(0.1, name="cell_dense1_dropout")(x)
    x = layers.Dense(128, activation="tanh", name="cell_dense2")(x)
    x = layers.Dropout(0.1, name="cell_dense2_dropout")(x)

    x = GraphConvBlock(256, step_num=1, name="cell_gc1")([x, cell_adj_input])
    x = GraphConvBlock(256, step_num=1, name="cell_gc2")([x, cell_adj_input])
    x = GraphConvBlock(256, step_num=1, name="cell_gc3")([x, cell_adj_input])
    x = GraphConvBlock(256, step_num=1, name="cell_gc4")([x, cell_adj_input])

    x = layers.GlobalAveragePooling1D(name="cell_pool")(x)

    cell_model = keras.Model(
        inputs=[cell_feat_input, cell_adj_input],
        outputs=x,
        name="cell_gcn",
    )

    return cell_model


def _create_drug_channel(drug_dim: int) -> keras.Model:
    """"""
    drug_feat_input = keras.Input(
        shape=(None, drug_dim), name="drug_feat_input"
    )
    drug_adj_input = keras.Input(shape=(None, None), name="drug_adj_input")

    x = [drug_feat_input, drug_adj_input]
    x = GraphConvBlock(units=256, step_num=1, name="drug_gc1")(x)
    x = [drug_feat_input, drug_adj_input]
    x = GraphConvBlock(units=128, step_num=1, name="drug_gc2")(x)
    x = layers.GlobalAveragePooling1D(name="drug_pool")(x)

    drug_model = keras.Model(
        inputs=[drug_feat_input, drug_adj_input],
        outputs=x,
        name="drug_gcn",
    )

    return drug_model


def create_model(
    cell_dim: int, drug_dim: int, cell_norm: layers.Normalization | None = None
) -> keras.Model:
    """"""
    cell_model = _create_cell_channel(cell_dim, cell_norm)
    drug_model = _create_drug_channel(drug_dim)

    channel_inputs = [*cell_model.input, *drug_model.input]
    channel_outputs = [cell_model.output, drug_model.output]

    x = layers.Concatenate(name="concat")(channel_outputs)
    x = layers.Dense(256, activation="tanh")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="tanh")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation="tanh")(x)
    output = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=channel_inputs, outputs=output, name="dualgcn")

    return model
