"""
DualGCN training/evaluation pipeline.
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

from cdrpy.models import dualgcn
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

    mol_path = paths.dualgcn.mol
    exp_path = paths.dualgcn.exp
    cnv_path = paths.dualgcn.cnv
    ppi_path = paths.dualgcn.ppi

    drug_encoders = dualgcn.load_drug_features(mol_path=mol_path)
    cell_encoders = dualgcn.load_cell_features(
        exp_path=exp_path, cnv_path=cnv_path, ppi_path=ppi_path
    )

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

    Assumes the first dataset provided is the training set.

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
    """Builds the DualGCN model."""
    params = cfg.model

    # extract shape from cell feature encoder
    cell_feat_enc = train_dataset.cell_encoders[0]
    cell_feat_norm = keras.layers.Normalization(axis=(1, 2))
    cell_feat_norm.adapt(
        np.array(cell_feat_enc.encode(train_dataset.cell_ids))
    )
    cell_dim = cell_feat_enc.shape[-1]

    # extract shape from drug feature encoder
    drug_feat_enc = train_dataset.drug_encoders[0]
    drug_dim = drug_feat_enc.shape[-1]

    model = dualgcn.create_model(cell_dim, drug_dim, cell_feat_norm)

    return model


def model_trainer(
    cfg: DictConfig,
    model: keras.Model,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> keras.Model:
    """Trains the DualGCN model.

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

    model = dualgcn.train(
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
    """Evaluates the DualGCN Model.

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
