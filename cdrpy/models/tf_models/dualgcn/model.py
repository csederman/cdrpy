"""

"""

from __future__ import annotations

import typing as t
import tensorflow as tf

keras = tf.keras  # pylance issue #1066

from keras import layers

from . import layers as L


class KerasMultiSourceDualGCNModel(object):
    def __init__(self, use_gexpr=True, use_cn=True, regr=True):  #
        self.use_gexpr = use_gexpr
        self.use_cn = use_cn
        self.regr = regr

    def create_model(self, drug_dim, cell_line_dim, units_list):
        # drug-graph input layer
        drug_feat_input = keras.Input(
            shape=(None, drug_dim), name="drug_feat_input"
        )
        drug_adj_input = keras.Input(shape=(None, None), name="drug_adj_input")

        # bio-graph input layer
        cell_line_feat_input = keras.Input(
            shape=(None, cell_line_dim), name="cell_line_feat_input"
        )
        cell_line_adj_input = keras.Input(
            shape=(None, None), name="cell_line_adj_input"
        )

        # drug-GCN
        GCN_layer = L.DualGCNGraphConv(
            units=units_list[0], step_num=1, name="DrugGraph_1_GCN"
        )([drug_feat_input, drug_adj_input])
        GCN_layer = layers.Activation("relu")(GCN_layer)
        GCN_layer = layers.BatchNormalization()(GCN_layer)
        GCN_layer = layers.Dropout(0.1, name="DrugGraph_1_out")(GCN_layer)

        GCN_layer = L.DualGCNGraphConv(
            units=128, step_num=1, name="DrugGraph_last_GCN"
        )([GCN_layer, drug_adj_input])
        GCN_layer = layers.Activation("relu")(GCN_layer)
        GCN_layer = layers.BatchNormalization()(GCN_layer)
        GCN_layer = layers.Dropout(0.1, name="DrugGraph_last_out")(GCN_layer)

        x_drug = layers.GlobalAveragePooling1D(name="DrugGraph_out")(GCN_layer)

        # bio-graph GCN
        cell_line_feat_input_high = layers.Dense(32, activation="tanh")(
            cell_line_feat_input
        )
        cell_line_feat_input_high = layers.Dropout(0.1)(
            cell_line_feat_input_high
        )
        cell_line_feat_input_high = layers.Dense(128, activation="tanh")(
            cell_line_feat_input_high
        )
        cell_line_feat_input_high = layers.Dropout(0.1)(
            cell_line_feat_input_high
        )

        cell_line_GCN = L.DualGCNGraphConv(
            units=256, step_num=1, name="CelllineGraph_1_GCN"
        )([cell_line_feat_input_high, cell_line_adj_input])
        cell_line_GCN = layers.Activation("relu")(cell_line_GCN)
        cell_line_GCN = layers.BatchNormalization()(cell_line_GCN)
        cell_line_GCN = layers.Dropout(0.1)(cell_line_GCN)

        cell_line_GCN = L.DualGCNGraphConv(units=256, step_num=1)(
            [cell_line_GCN, cell_line_adj_input]
        )
        cell_line_GCN = layers.Activation("relu")(cell_line_GCN)
        cell_line_GCN = layers.BatchNormalization()(cell_line_GCN)
        cell_line_GCN = layers.Dropout(0.1)(cell_line_GCN)

        cell_line_GCN = L.DualGCNGraphConv(units=256, step_num=1)(
            [cell_line_GCN, cell_line_adj_input]
        )
        cell_line_GCN = layers.Activation("relu")(cell_line_GCN)
        cell_line_GCN = layers.BatchNormalization()(cell_line_GCN)
        cell_line_GCN = layers.Dropout(0.1)(cell_line_GCN)

        cell_line_GCN = L.DualGCNGraphConv(units=256, step_num=1)(
            [cell_line_GCN, cell_line_adj_input]
        )
        cell_line_GCN = layers.Activation("relu")(cell_line_GCN)
        cell_line_GCN = layers.BatchNormalization()(cell_line_GCN)
        cell_line_GCN = layers.Dropout(0.1)(cell_line_GCN)

        x_cell_line = layers.GlobalAveragePooling1D(name="CelllineGraph_out")(
            cell_line_GCN
        )

        x = layers.Concatenate(name="Merge_Drug_Cellline_graphs")(
            [x_cell_line, x_drug]
        )
        x = layers.Dense(256, activation="tanh")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="tanh")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(10, activation="tanh")(x)
        output = layers.Dense(1, name="output")(x)

        model = keras.Model(
            inputs=[
                drug_feat_input,
                drug_adj_input,
                cell_line_feat_input,
                cell_line_adj_input,
            ],
            outputs=output,
        )

        return model
