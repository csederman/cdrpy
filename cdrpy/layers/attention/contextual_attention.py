"""

"""

from __future__ import annotations

import tensorflow as tf

from tensorflow import keras

from .. import util as L


class ContextualAttention(keras.layers.Layer):
    """Contextual attention layer.

    This is a reimplementation of contextual attention proposed in [1] and [2]:
        https://github.com/PaccMann/paccmann_predictor_tf

    Inspired by Bahdanau attention, this layer defines how well each token of
    the encoded SMILES string targets the provided genes.

    Parameters
    ----------
        attention_size: The number of attention units.
        hidden_size: The number of hidden units.

    References
    ----------
    .. [1] Oskooei, Ali and Born, Jannis and Manica, Matteo and Subramanian,
           Vigneshwari and Saez-Rodriguez, Julio and Martinez, Maria Rodriguez.
           PaccMann: Prediction of anticancer compound sensitivity with
           multi-modal attention-based neural network. arXiv preprint
           arXiv:1811.06802, 2018.

    .. [2] Manica, Matteo and Oskooei, Ali and Born, Jannis and Subramanian,
           Vigneshwari and Saez-Rodriguez, Julio and Rodriguez Martinez, Maria.
           Toward Explainable Anticancer Compound Sensitivity Prediction via
           Multimodal Attention-Based Convolutional Encoders. Molecular
           Pharmaceutics, 2019.
    """

    def __init__(
        self,
        attention_size: int,
        hidden_size: int,
        num_genes: int,
        num_gene_features: int,
    ) -> None:
        self.w_num_gene_features = tf.Variable(
            tf.random.normal([num_gene_features], stddev=0.1)
        )
        self.w_genes = tf.Variable(
            tf.random.normal([num_genes, attention_size], stddev=0.1)
        )
        self.b_genes = tf.Variable(
            tf.random.normal([attention_size], stddev=0.1)
        )
        self.w_smiles = tf.Variable(
            tf.random.normal([hidden_size, attention_size], stddev=0.1)
        )
        self.b_smiles = tf.Variable(
            tf.random.normal([attention_size], stddev=0.1)
        )
        self.v = tf.Variable(tf.random.normal([attention_size], stddev=0.1))

        self.tensordot_1 = L.TensorDot(axis=[2, 0])
        self.tensordot_2 = L.TensorDot(axis=1)
        self.reduce_sum = L.ReduceSum(axis=1)
        self.softmax = keras.layers.Softmax()

        self.expand_dim_1 = L.ExpandDimLayer(axis=2)
        self.expand_dim_2 = L.ExpandDimLayer(axis=1)
        self.expand_dim_3 = L.ExpandDimLayer(axis=-1)

    def call(self, inputs):
        """"""
        genes = (
            self.expand_dim_1(inputs[0])
            if len(inputs[0].shape) == 2
            else inputs[0]
        )
        genes_collapsed = self.tensor_dot_1([genes, self.w_num_gene_features])

        x = self.tensor_dot_2([genes_collapsed, self.w_genes])
        x = x + self.b_genes
        x = self.expand_dim_2(x)

        y = self.tensor_dot_2([inputs[1], self.w_smiles])
        y = y + self.b_smiles

        x = x + y
        x = keras.activations.tanh(x)

        xv = self.tensor_dot_2([x, self.v])
        alphas = self.softmax(xv)

        out = self.expand_dim_3(alphas)
        out = inputs[1] * out

        return self.reduce_sum(out)
