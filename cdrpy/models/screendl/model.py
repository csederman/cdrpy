"""

"""

from __future__ import annotations

import tensorflow as tf

from tensorflow import keras
from keras import Model
from keras import layers


def _create_expression_subnetwork(
    dim: int,
    norm_layer: layers.Normalization | None = None,
    hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the expression subnetwork."""
    x = input_layer = layers.Input((dim,), name="exp_input")

    if norm_layer is not None:
        if not norm_layer.is_adapted:
            # FIXME: change this to a warning since you can still adapt later
            # FIXME: this needs to be decoupled from model creation - I need to
            #   refactor the models so normalization occurs during data
            #   preprocessing.
            raise ValueError("requires adapted normalization layer...")
        x = norm_layer(x)

    if hidden_dims is None:
        # FIXME: Decide where I want batch normalization and dropout
        #   -> Before or after the first dense layer?
        #   -> I might want it before if I don't include the initial dense layer
        #      with the same dim as the input because then it would drop out
        #      random gene inputs during uncertainty quantification which might
        #      be a desired property.
        # hidden_dims = [dim, dim // 2, dim // 4, dim // 8]
        hidden_dims = [dim // 2, dim // 4, dim // 8]

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 1):
        # NOTE: use tanh activation for the embedding output
        act = "tanh" if i == n_hidden else "relu"
        x = layers.Dense(units, activation=act, name=f"exp_dn_{i}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"exp_bn_{i}")(x)
        if use_dropout:
            x = layers.Dropout(dropout_rate, name=f"exp_dr_{i}")(x)

    return Model(inputs=input_layer, outputs=x, name="exp_subnet")


def _create_mutation_subnetwork(
    dim: int,
    hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the mutation subnetwork."""
    x = input_layer = layers.Input((dim,), name="mut_input")

    if hidden_dims is None:
        # hidden_dims = [dim, dim // 2, dim // 4, dim // 8]
        hidden_dims = [dim // 2, dim // 4, dim // 8]

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 1):
        # NOTE: use tanh activation for the embedding output
        act = "tanh" if i == n_hidden else "relu"
        x = layers.Dense(units, activation=act, name=f"mut_dn_{i}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"mut_bn_{i}")(x)
        if use_dropout:
            x = layers.Dropout(dropout_rate, name=f"mut_dr_{i}")(x)

    return Model(inputs=input_layer, outputs=x, name="mut_subnet")


def _create_copy_number_subnetwork(
    dim: int,
    norm_layer: layers.Normalization | None = None,
    hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the mutation subnetwork."""
    x = input_layer = layers.Input((dim,), name="cnv_input")

    if norm_layer is not None:
        if not norm_layer.is_adapted:
            # FIXME: change this to a warning since you can still adapt later
            # FIXME: this needs to be decoupled from model creation - I need to
            #   refactor the models so normalization occurs during data
            #   preprocessing.
            raise ValueError("requires adapted normalization layer...")
        x = norm_layer(x)

    if hidden_dims is None:
        # hidden_dims = [dim, dim // 2, dim // 4, dim // 8]
        hidden_dims = [dim // 2, dim // 4, dim // 8]

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 1):
        # NOTE: Use tanh activation for the embedding output
        act = "tanh" if i == n_hidden else "relu"
        x = layers.Dense(units, activation=act, name=f"cnv_dn_{i}")(x)
        # FIXME: make sure batch norm should come before dropout
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"cnv_bn_{i}")(x)
        if use_dropout:
            x = layers.Dropout(dropout_rate, name=f"cnv_dr_{i}")(x)

    return Model(inputs=input_layer, outputs=x, name="cnv_subnet")


def _create_cell_subnetwork(
    exp_dim: int,
    mut_dim: int | None = None,
    cnv_dim: int | None = None,
    exp_norm_layer: layers.Normalization | None = None,
    cnv_norm_layer: layers.Normalization | None = None,
    exp_hidden_dims: list[int] | None = None,
    mut_hidden_dims: list[int] | None = None,
    cnv_hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the cell subnetwork.

    Parameters
    ----------
        exp_dim: The dimension of the gene expression subnetwork.
        mut_dim: The dimension of the mutation subnetwork.
        cnv_dim: The dimension of the copy number subnetwork.
        exp_norm_layer: An optional `keras.layers.Normalization` layer.
        cnv_norm_layer: An optional `keras.layers.Normalization` layer.
        exp_hidden_dims: An optional list specifying hidden layers/units for
            the gene expression subnetwork.
        mut_hidden_dims: An optional list specifying hidden layers/units for
            the mutation subnetwork.
        cnv_hidden_dims: An optional list specifying hidden layers/units for
            the copy number subnetwork.
        use_batch_norm: Whether or not to use batch normalization.
        use_dropout: Whether or not to use dropout.
        dropout_rate: The dropout rate. Ignored if `use_dropout` is `False`.

    Returns
    -------
        The drug subnetwork `keras.Model` instance.
    """
    exp_subnet = _create_expression_subnetwork(
        exp_dim,
        exp_norm_layer,
        exp_hidden_dims,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
    )
    subnet_inputs = [exp_subnet.input]
    subnet_output = exp_subnet.output

    if mut_dim is not None:
        mut_subnet = _create_mutation_subnetwork(
            mut_dim,
            mut_hidden_dims,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        subnet_inputs.append(mut_subnet.input)
        subnet_output = layers.Concatenate()(
            [subnet_output, mut_subnet.output]
        )

    if cnv_dim is not None:
        cnv_subnet = _create_copy_number_subnetwork(
            cnv_dim,
            cnv_norm_layer,
            cnv_hidden_dims,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        subnet_inputs.append(cnv_subnet.input)
        subnet_output = layers.Concatenate()(
            [subnet_output, cnv_subnet.output]
        )

    return Model(
        inputs=subnet_inputs, outputs=subnet_output, name="cell_subnet"
    )


def _create_drug_subnetwork(
    mol_dim: int,
    mol_hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
) -> keras.Sequential:
    """Creates the drug subnetwork.

    Parameters
    ----------
        mol_dim: The dimension of the drug subnetwork.
        mol_hidden_dims: Optional list specifying hidden layers/units.
        use_batch_norm: Whether or not to use batch normalization.
        use_dropout: Whether or not to use dropout.
        dropout_rate: The dropout rate. Ignored if `use_dropout` is `False`.

    Returns
    -------
        The drug subnetwork `keras.Model` instance.
    """
    x = input_layer = layers.Input((mol_dim,), name="fp_input")

    if mol_hidden_dims is None:
        # mol_hidden_dims = [mol_dim, mol_dim // 2, mol_dim // 4, mol_dim // 8]
        mol_hidden_dims = [mol_dim // 2, mol_dim // 4, mol_dim // 8]

    n_hidden = len(mol_hidden_dims)
    for i, units in enumerate(mol_hidden_dims, 1):
        # NOTE: Use tanh activation for the embedding output
        act = "tanh" if i == n_hidden else "relu"
        x = layers.Dense(units, activation=act, name=f"drug_dn_{i}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"drug_bn_{i}")(x)
        if use_dropout:
            x = layers.Dropout(dropout_rate, name=f"drug_dr_{i}")(x)

    return Model(inputs=input_layer, outputs=x, name="drug_subnet")


def create_model(
    exp_dim: int,
    mol_dim: int,
    mut_dim: int | None = None,
    cnv_dim: int | None = None,
    exp_norm_layer: layers.Normalization | None = None,
    cnv_norm_layer: layers.Normalization | None = None,
    exp_hidden_dims: list[int] | None = None,
    mut_hidden_dims: list[int] | None = None,
    cnv_hidden_dims: list[int] | None = None,
    mol_hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """"""
    cell_subnet = _create_cell_subnetwork(
        exp_dim,
        mut_dim,
        cnv_dim,
        exp_norm_layer=exp_norm_layer,
        cnv_norm_layer=cnv_norm_layer,
        exp_hidden_dims=exp_hidden_dims,
        mut_hidden_dims=mut_hidden_dims,
        cnv_hidden_dims=cnv_hidden_dims,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
    )

    drug_subnet = _create_drug_subnetwork(
        mol_dim,
        mol_hidden_dims=mol_hidden_dims,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
    )

    if isinstance(cell_subnet.input, list):
        # FIXME: Should probably not concatenate here and pass the outputs of
        #   the cell encoders and drug encoders separately.
        subnet_inputs = [*cell_subnet.input, drug_subnet.input]
    else:
        subnet_inputs = [cell_subnet.input, drug_subnet.input]
    subnet_outputs = [cell_subnet.output, drug_subnet.output]

    latent_dim = sum(
        (cell_subnet.output_shape[1], drug_subnet.output_shape[1])
    )
    hidden_dims = [
        latent_dim,
        latent_dim // 2,
        latent_dim // 4,
        latent_dim // 8,
        latent_dim // 10,
    ]

    x = layers.Concatenate(name="concat")(subnet_outputs)
    for i, units in enumerate(hidden_dims, 1):
        x = layers.Dense(units, activation="relu", name=f"shared_dn_{i}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"shared_bn_{i}")(x)
        if use_dropout:
            x = layers.Dropout(dropout_rate, name=f"shared_dr_{i}")(x)
    output = layers.Dense(1, "linear", name="final_act")(x)

    return Model(inputs=subnet_inputs, outputs=output, name="ScreenDL")
