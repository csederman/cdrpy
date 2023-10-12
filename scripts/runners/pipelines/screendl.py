"""
Run utilities for ScreenDL.
"""

from __future__ import annotations

import os
import logging

import numpy as np
import tensorflow as tf
import typing as t

from omegaconf import DictConfig
from pathlib import Path
from tensorflow import keras

from cdrpy.models import screendl
from cdrpy.data.datasets import Dataset, get_predictions
from cdrpy.data.preprocess import normalize_responses
from cdrpy.splits import load_split


if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import PandasEncoder


log = logging.getLogger(__name__)


def data_loader(cfg: DictConfig) -> Dataset:
    """Loads the input dataset.

    Parameters
    ----------
        cfg:

    Returns
    -------
    """
    paths = cfg.dataset.sources

    mol_path = paths.screendl.mol
    exp_path = paths.screendl.exp
    mut_path = paths.screendl.mut if cfg.model.feat.use_mut else None
    cnv_path = paths.screendl.cnv if cfg.model.feat.use_cnv else None
    ont_path = paths.screendl.ont if cfg.model.feat.use_ont else None

    drug_encoders = [screendl.load_drug_features(mol_path)]
    cell_encoders = screendl.load_cell_features(
        exp_path=exp_path,
        mut_path=mut_path,
        cnv_path=cnv_path,
        ont_path=ont_path,
    )
    cell_encoders = list(filter(None, cell_encoders))

    return Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
    )


def data_splitter(
    cfg: DictConfig, dataset: Dataset
) -> tuple[Dataset, Dataset, Dataset]:
    """Splits the dataset into train/validation/test sets.

    Parameters
    ----------
        cfg:
        dataset:

    Returns
    -------
    """
    split_id = cfg.dataset.split.id
    split_dir = cfg.dataset.split.dir
    split_name = cfg.dataset.split.name
    split_path = os.path.join(split_dir, split_name)

    split = load_split(split_path, split_id)

    return (
        dataset.select(split.train_ids, name="train"),
        dataset.select(split.val_ids, name="val"),
        dataset.select(split.test_ids, name="test"),
    )


def data_preprocessor(
    cfg: DictConfig,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Preprocessing pipeline.

    Parameters
    ----------
        cfg:
        datasets:

    Returns
    -------
        A (train, validation, test) tuple of processed datasets.
    """
    train_dataset, val_dataset, test_dataset = normalize_responses(
        train_dataset,
        val_dataset,
        test_dataset,
        norm_method=cfg.dataset.preprocess.norm,
    )
    return train_dataset, val_dataset, test_dataset


def model_builder(cfg: DictConfig, train_dataset: Dataset) -> keras.Model:
    """Builds the ScreenDL model."""
    params = cfg.model

    # extract exp shape from cell encoders
    exp_enc: PandasEncoder = train_dataset.cell_encoders[0]
    exp_dim = exp_enc.shape[-1]
    exp_norm_layer = keras.layers.Normalization(name="exp_norm")
    exp_norm_layer.adapt(np.array(exp_enc.encode(train_dataset.cell_ids)))

    # extract mut shape from cell encoders
    mut_dim = None
    if params.feat.use_mut:
        mut_enc: PandasEncoder = train_dataset.cell_encoders[1]
        mut_dim = mut_enc.shape[-1]

    # extract cnv shape from cell encoders
    cnv_dim = cnv_norm_layer = None
    if params.feat.use_cnv:
        cnv_enc: PandasEncoder = train_dataset.cell_encoders[-1]
        cnv_dim = cnv_enc.shape[-1]
        cnv_norm_layer = keras.layers.Normalization(name="cnv_norm")
        cnv_norm_layer.adapt(np.array(cnv_enc.encode(train_dataset.cell_ids)))

    # extract tissue ontology shape from cell encoders
    ont_dim = None
    if params.feat.use_ont:
        # FIXME: need to change to using an ordered dict for the cell encoders
        #   -> then I can extract the encoders using the keys instead of the index
        ont_enc: PandasEncoder = train_dataset.cell_encoders[-1]
        ont_dim = ont_enc.shape[-1]

    # extract mol shape from drug encoders
    mol_enc: PandasEncoder = train_dataset.drug_encoders[0]
    mol_dim = mol_enc.shape[-1]

    model = screendl.create_model(
        exp_dim,
        mol_dim,
        mut_dim,
        cnv_dim,
        ont_dim,
        exp_norm_layer=exp_norm_layer,
        cnv_norm_layer=cnv_norm_layer,
        exp_hidden_dims=params.hyper.hidden_dims.exp,
        mut_hidden_dims=params.hyper.hidden_dims.mut,
        cnv_hidden_dims=params.hyper.hidden_dims.cnv,
        ont_hidden_dims=params.hyper.hidden_dims.ont,
        mol_hidden_dims=params.hyper.hidden_dims.mol,
        use_batch_norm=params.hyper.use_batch_norm,
        use_dropout=params.hyper.use_dropout,
        dropout_rate=params.hyper.dropout_rate,
        initial_dropout_rate=params.hyper.initial_dropout_rate,
    )

    return model


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> keras.Model:
    """Trains the ScreenDL model.

    Parameters
    ----------
        cfg:
        model:
        train_dataset:
        val_dataset:

    Returns
    -------
        The trained `keras.Model` instance.
    """
    params = cfg.model
    opt = keras.optimizers.Adam(learning_rate=params.hyper.learning_rate)

    model = screendl.train(
        model,
        opt,
        train_dataset,
        val_dataset,
        batch_size=params.hyper.batch_size,
        epochs=params.hyper.epochs,
        save_dir=("." if params.io.save is True else None),
        log_dir=("./logs" if params.io.tensorboard is True else None),
        early_stopping=params.hyper.early_stopping,
        tensorboard=params.io.tensorboard,
    )

    return model


def model_evaluator(
    cfg: DictConfig,
    model: keras.Model,
    datasets: t.Iterable[Dataset],
) -> None:
    """Evaluates the ScreenDL Model.

    Parameters
    ----------
        cfg:
        model:
        datasets:
    """
    pred_df = get_predictions(datasets, model, fold=cfg.dataset.split.id)
    pred_df.to_csv("predictions.csv", index=False)

    if cfg.dataset.output.save:
        root_dir = Path("./datasets")
        root_dir.mkdir()
        for ds in datasets:
            subdir = root_dir / str(ds.name)
            subdir.mkdir()
            ds.save(subdir)


def run_pipeline(cfg: DictConfig) -> None:
    """Runs the ScreenDL training pipeline."""
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info(f"Loading {dataset_name}...")
    dataset = data_loader(cfg)

    log.info(f"Splitting {dataset_name}...")
    train_dataset, val_dataset, test_dataset = data_splitter(cfg, dataset)

    log.info(f"Preprocessing {dataset_name}...")
    train_dataset, val_dataset, test_dataset = data_preprocessor(
        cfg, train_dataset, val_dataset, test_dataset
    )

    log.info(f"Building {model_name}...")
    model = model_builder(cfg, train_dataset)

    log.info(f"Training {model_name}...")
    model = model_trainer(cfg, model, train_dataset, val_dataset)

    log.info(f"Evaluating {model_name}...")
    model_evaluator(cfg, model, [train_dataset, val_dataset, test_dataset])
