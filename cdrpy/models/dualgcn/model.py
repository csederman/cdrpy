"""

"""

from __future__ import annotations

import typing as t
import tensorflow as tf

from tensorflow import keras
from keras import layers

from cdrpy.layers.graph import GraphConvBlock


def _create_cell_subnetwork(
    cell_dim: int, cell_norm: layers.Normalization | None = None
) -> keras.Model:
    """"""
    cell_feat_input = keras.Input(
        shape=(None, cell_dim), name="cell_feat_input"
    )
    cell_adj_input = keras.Input(shape=(None, None), name="cell_adj_input")

    if cell_norm is not None:
        cell_feat_input = cell_norm(cell_feat_input)
    x = layers.Dense(32, activation="tanh", name="cell_dn_1")(cell_feat_input)
    x = layers.Dropout(0.1, name="cell_dr_1")(x)
    x = layers.Dense(128, activation="tanh", name="cell_dn_2")(x)
    x = layers.Dropout(0.1, name="cell_dr_2")(x)

    x = GraphConvBlock(256, step_num=1, name="cell_gc_1")([x, cell_adj_input])
    x = GraphConvBlock(256, step_num=1, name="cell_gc_2")([x, cell_adj_input])
    x = GraphConvBlock(256, step_num=1, name="cell_gc_3")([x, cell_adj_input])
    x = GraphConvBlock(256, step_num=1, name="cell_gc_4")([x, cell_adj_input])

    x = layers.GlobalAveragePooling1D(name="cell_gap_1")(x)

    cell_model = keras.Model(
        inputs=[cell_feat_input, cell_adj_input], outputs=x, name="cell_subnet"
    )

    return cell_model


def _create_drug_subnetwork(drug_dim: int) -> keras.Model:
    """"""
    drug_feat_input = keras.Input(
        shape=(None, drug_dim), name="drug_feat_input"
    )
    drug_adj_input = keras.Input(shape=(None, None), name="drug_adj_input")

    x = [drug_feat_input, drug_adj_input]
    x = GraphConvBlock(units=256, step_num=1, name="drug_gc_1")(x)
    x = [x, drug_adj_input]
    x = GraphConvBlock(units=128, step_num=1, name="drug_gc_2")(x)
    x = layers.GlobalAveragePooling1D(name="drug_pool")(x)

    drug_model = keras.Model(
        inputs=[drug_feat_input, drug_adj_input], outputs=x, name="drug_subnet"
    )

    return drug_model


def create_model(
    cell_dim: int, drug_dim: int, cell_norm: layers.Normalization | None = None
) -> keras.Model:
    """"""
    cell_subnetwork = _create_cell_subnetwork(cell_dim, cell_norm)
    drug_subnetwork = _create_drug_subnetwork(drug_dim)

    subnetwork_inputs = [*cell_subnetwork.input, *drug_subnetwork.input]
    subnetwork_outputs = [cell_subnetwork.output, drug_subnetwork.output]

    x = layers.Concatenate(name="concat")(subnetwork_outputs)
    x = layers.Dense(256, activation="tanh")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="tanh")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation="tanh")(x)
    output = layers.Dense(1, name="output")(x)

    model = keras.Model(
        inputs=subnetwork_inputs, outputs=output, name="DualGCN"
    )

    return model
