"""
Trains and evaluates the ScreenDL model on a single fold.
"""


from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import random

import numpy as np
import tensorflow as tf
import typing as t

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from omegaconf import DictConfig
from tensorflow import keras

from cdrpy.models import screendl
from cdrpy.data.datasets import Dataset, get_predictions
from cdrpy.data.preprocess import normalize_responses
from cdrpy.splits import load_split

if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import PandasEncoder


def get_data(cfg: DictConfig) -> tuple[Dataset, Dataset, Dataset]:
    """Loads the input data."""
    paths = cfg.dataset.sources
    split_path = os.path.join(cfg.dataset.split.dir, cfg.dataset.split.name)

    cell_enc = screendl.load_cell_features(paths.screendl.exp)
    drug_enc = screendl.load_drug_features(paths.screendl.mol)

    dataset = Dataset.from_csv(
        paths.labels,
        name=cfg.dataset.name,
        cell_encoders=[cell_enc],
        drug_encoders=[drug_enc],
    )

    split = load_split(split_path, cfg.dataset.split.id)

    train_ds = dataset.select(split.train_ids, name="train")
    val_ds = dataset.select(split.val_ids, name="val")
    test_ds = dataset.select(split.test_ids, name="test")

    train_ds, val_ds, test_ds = normalize_responses(
        train_ds, val_ds, test_ds, norm_method=cfg.dataset.preprocess.norm
    )

    return (train_ds, val_ds, test_ds)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """Train and evaluate ScreenDL on a given dataset and fold."""
    params = cfg.model

    train_ds, val_ds, test_ds = get_data(cfg)
    cell_enc: PandasEncoder = train_ds.cell_encoders[0]
    drug_enc: PandasEncoder = train_ds.drug_encoders[0]

    # NOTE: downsample for testing
    # train_ds = train_ds.sample(1000, name="train")
    # val_ds = val_ds.sample(1000, name="val")
    # test_ds = test_ds.sample(1000, name="test")

    cell_norm = keras.layers.Normalization()
    cell_norm.adapt(np.array(cell_enc.encode(train_ds.cell_ids)))

    cell_dim = cell_enc.shape[-1]
    drug_dim = drug_enc.shape[-1]

    opt = keras.optimizers.Adam(learning_rate=params.learning_rate)
    model = screendl.create_model(cell_dim, drug_dim, cell_norm)

    save_dir = "." if params.save is True else None
    log_dir = "./logs" if params.tensorboard is True else None

    model = screendl.train(
        model,
        opt,
        train_ds,
        val_ds,
        batch_size=params.batch_size,
        epochs=params.epochs,
        save_dir=save_dir,
        log_dir=log_dir,
        early_stopping=params.early_stopping,
        tensorboard=params.tensorboard,
    )

    pred_df = get_predictions(
        (train_ds, val_ds, test_ds), model, fold=cfg.dataset.split.id
    )
    pred_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    run()
