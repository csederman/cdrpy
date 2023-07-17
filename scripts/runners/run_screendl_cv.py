#!/usr/bin/env python
"""

"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import random
import logging

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

keras = tf.keras  # pylance issue #1066

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from dataclasses import dataclass
from pathlib import Path
from hydra.core.config_store import ConfigStore
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cdrpy.data import _datasets
from cdrpy.splits import Split
from cdrpy.trans.transformers import GroupedStandardScaler
from cdrpy.util.io import read_pickled_list
from cdrpy.metrics.tf_metrics import pearson
from cdrpy.models.tf_models import screendl
from cdrpy.conf import Config


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def load_splits(split_dir: Path) -> t.Generator[Split, None, None]:
    """"""
    split_dir = Path(split_dir)
    for i in range(10):
        train_ids = read_pickled_list(split_dir / f"train_{i+1}.pickle")
        val_ids = read_pickled_list(split_dir / f"val_{i+1}.pickle")
        test_ids = read_pickled_list(split_dir / f"test_{i+1}.pickle")
        yield Split(train_ids, val_ids, test_ids)


def scale_expression(
    train_ds: _datasets.Dataset,
    val_ds: _datasets.Dataset,
    test_ds: _datasets.Dataset,
    feature_name: str = "rna",
) -> tuple[_datasets.Dataset, _datasets.Dataset, _datasets.Dataset]:
    """"""
    exp_normalizer = MinMaxScaler()
    _ = exp_normalizer.fit(train_ds.cell_features[feature_name].values)
    exp_norm_func = _datasets.make_pandas_transformer(exp_normalizer.transform)

    train_ds = train_ds.transform_cell_feature(feature_name, exp_norm_func)
    val_ds = val_ds.transform_cell_feature(feature_name, exp_norm_func)
    test_ds = test_ds.transform_cell_feature(feature_name, exp_norm_func)

    return (train_ds, val_ds, test_ds)


def scale_labels(
    train_ds: _datasets.Dataset,
    val_ds: _datasets.Dataset,
    test_ds: _datasets.Dataset,
) -> tuple[_datasets.Dataset, _datasets.Dataset, _datasets.Dataset]:
    """"""
    train_labels = train_ds.labels.reshape(-1, 1)

    label_scaler = StandardScaler()
    _ = label_scaler.fit(train_labels)

    def transform(ds: _datasets.Dataset) -> _datasets.Dataset:
        labels = ds.labels.reshape(-1, 1)
        ds.obs[["label"]] = label_scaler.transform(labels)
        return ds

    return (transform(train_ds), transform(val_ds), transform(test_ds))


def scale_labels_grouped(
    train_ds: _datasets.Dataset,
    val_ds: _datasets.Dataset,
    test_ds: _datasets.Dataset,
) -> tuple[_datasets.Dataset, _datasets.Dataset, _datasets.Dataset]:
    """"""
    train_labels = train_ds.labels.reshape(-1, 1)
    train_groups = train_ds.drug_ids

    label_scaler = GroupedStandardScaler(use_median=True)
    _ = label_scaler.fit(train_labels, groups=train_groups)

    def transform(ds: _datasets.Dataset) -> _datasets.Dataset:
        labels = ds.labels.reshape(-1, 1)
        groups = ds.drug_ids
        ds.obs[["label"]] = label_scaler.transform(labels, groups=groups)
        return ds

    return (transform(train_ds), transform(val_ds), transform(test_ds))


def get_pred_df(ds: _datasets.Dataset, preds: np.ndarray) -> pd.DataFrame:
    """"""
    return pd.DataFrame(
        {
            "cell_id": ds.cell_ids,
            "drug_id": ds.drug_ids,
            "label": ds.labels,
            "pred": preds,
            "set": ds.name,
        }
    )


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


def preprocess_data(
    ds: _datasets.Dataset,
    split: Split,
    scale_labels_per_drug: bool,
) -> tuple[_datasets.Dataset, _datasets.Dataset, _datasets.Dataset]:
    """"""
    train_ds = ds.select(ids=split.train_ids, name="train")
    val_ds = ds.select(ids=split.val_ids, name="val")
    test_ds = ds.select(ids=split.test_ids, name="test")

    train_ds, val_ds, test_ds = scale_expression(train_ds, val_ds, test_ds)
    if scale_labels_per_drug:
        train_ds, val_ds, test_ds = scale_labels_grouped(
            train_ds, val_ds, test_ds
        )
    else:
        train_ds, val_ds, test_ds = scale_labels(train_ds, val_ds, test_ds)

    return train_ds, val_ds, test_ds


def fit_model(
    train_ds: _datasets.Dataset,
    val_ds: _datasets.Dataset,
    out_dir: Path,
    log_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """
    FIXME: remove cell_features and drug_features args
    FIXME: make this more flexible
    """
    # cell_features = ["dna", "rna"]
    cell_features = ["rna"]
    drug_features = ["fp"]

    X_train, y_train, *_ = train_ds.encode(cell_features, drug_features)
    X_val, y_val, *_ = val_ds.encode(cell_features, drug_features)

    # dna_shape = X_train[0].shape[1]
    # rna_shape = X_train[1].shape[1]
    # fp_shape = X_train[2].shape[1]
    rna_shape = X_train[0].shape[1]
    fp_shape = X_train[1].shape[1]

    # dna_model = screendl._build_dna_model(dna_shape)
    rna_model = screendl._build_rna_model(rna_shape)
    fp_model = screendl._build_fp_model(fp_shape)

    model = screendl._build_model([rna_model, fp_model])
    # model = screendl._build_model([dna_model, rna_model, fp_model])

    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=20000,
        decay_rate=0.96,
        staircase=True,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=scheduler),
        loss=keras.losses.MeanSquaredError(),
        metrics=[pearson, "mse", "mae"],
    )

    board_cb = keras.callbacks.TensorBoard(str(log_dir), histogram_freq=1)
    stop_cb = keras.callbacks.EarlyStopping(
        "val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    hx = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[board_cb, stop_cb],
    )

    fit_history = pd.DataFrame(hx.history)
    fit_history.to_csv(out_dir / "fit_history.csv", index=False)

    return model


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="config",
)
def run(cfg: Config) -> None:
    """"""
    ds_dir = Path(cfg.dataset.dir)
    ds_sources = cfg.dataset.sources

    label_source = ds_dir / ds_sources.label
    cell_sources = {k: (ds_dir / v) for k, v in ds_sources.cell.items()}
    drug_sources = {k: (ds_dir / v) for k, v in ds_sources.drug.items()}

    ds = _datasets.load_dataset(
        label_source=label_source,
        label_args=cfg.dataset.label_args,
        cell_sources=cell_sources,
        drug_sources=drug_sources,
        name=cfg.dataset.name,
    )

    out_dir = Path(cfg.output.out_dir)
    out_dir.mkdir()

    log_dir = Path(cfg.output.log_dir)
    log_dir.mkdir()

    split_dir = ds_dir / cfg.dataset.split_subdir
    splits = load_splits(split_dir)
    for i, s in enumerate(splits):
        fold_out_dir = out_dir / f"fold_{i+1}"
        fold_out_dir.mkdir()

        fold_log_dir = log_dir / f"fold_{i+1}"
        fold_log_dir.mkdir()

        train_ds, val_ds, test_ds = preprocess_data(
            ds, s, cfg.dataset.scale_labels_per_drug
        )

        model = fit_model(
            train_ds,
            val_ds,
            out_dir=fold_out_dir,
            log_dir=fold_log_dir,
            epochs=cfg.model.epochs,
            batch_size=cfg.model.batch_size,
            learning_rate=cfg.model.learning_rate,
        )

        pred_dfs = []
        for d in (train_ds, val_ds, test_ds):
            for batch in d.encode_batches(["rna"], ["fp"], 32):
                # for batch in d.encode_batches(["dna", "rna"], ["fp"], 32):
                batch_features, batch_labels, batch_cells, batch_drugs = batch
                batch_preds = model.predict_on_batch(batch_features)
                batch_preds = batch_preds.reshape(-1)
                pred_df = pd.DataFrame(
                    {
                        "cell_id": batch_cells,
                        "drug_id": batch_drugs,
                        "y_true": batch_labels,
                        "y_pred": batch_preds,
                        "split": d.name,
                        "fold": i + 1,
                    }
                )
                pred_dfs.append(pred_df)

        pred_df = pd.concat(pred_dfs)
        pred_df.to_csv(fold_out_dir / f"predictions.csv", index=False)

        if cfg.output.save_model:
            model.save(fold_out_dir / "model")
            model.save_weights(fold_out_dir / "weights")


if __name__ == "__main__":
    run()
