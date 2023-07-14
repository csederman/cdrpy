"""

"""

from __future__ import annotations

import os
import copy

import numpy as np
import pandas as pd
import typing as t
import tensorflow as tf

from dataclasses import dataclass
from pathlib import Path

from ..util.types import PathLike


SourceDict = t.Dict[str, t.Union[PathLike, Path]]


def make_pandas_transformer(
    func: t.Callable[[np.ndarray], np.ndarray]
) -> t.Callable[[pd.DataFrame], pd.DataFrame]:
    """"""

    # NOTE: this assumes that the function does not reorder the dataframe
    def transformer(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            func(X.to_numpy()), index=X.index, columns=X.columns
        )

    return transformer


def load_features(
    **kwargs: dict[str, PathLike | Path]
) -> dict[str, pd.DataFrame]:
    """"""
    feature_dict = {}
    for feature_name, feature_source in kwargs.items():
        feature_dict[feature_name] = pd.read_csv(feature_source, index_col=0)
    return feature_dict


def load_labels(
    source: PathLike | Path,
    cell_id_col: str = "cell_id",
    drug_id_col: str = "drug_id",
    label_col: str = "label",
) -> pd.DataFrame:
    """"""
    column_mapping = {
        cell_id_col: "cell_id",
        drug_id_col: "drug_id",
        label_col: "label",
    }
    return pd.read_csv(source).rename(columns=column_mapping)


def load_dataset(
    label_source: PathLike | Path,
    label_args: dict[str, str],
    cell_sources: SourceDict,
    drug_sources: SourceDict,
    **kwargs,
) -> Dataset:
    """"""
    cell_features = load_features(**cell_sources)
    drug_features = load_features(**drug_sources)
    labels = load_labels(label_source, **label_args)
    return Dataset(labels, cell_features, drug_features, **kwargs)


@dataclass(repr=False)
class Dataset:
    """Cancer drug response prediction dataset.

    FIXME: add metadata for cells and drugs
    """

    obs: pd.DataFrame  # FIXME: add some sort of validator here
    cell_features: dict[str, pd.DataFrame]
    drug_features: dict[str, pd.DataFrame]
    name: str | None = None
    desc: str | None = None

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name}, size={self.size:_}, "
            f"n_cells={self.n_cells:_}, n_drugs={self.n_drugs:_})"
        )

    @property
    def labels(self) -> np.ndarray:
        return self.obs["label"].to_numpy()

    @property
    def cell_ids(self) -> np.ndarray:
        return self.obs["cell_id"].to_numpy()

    @property
    def drug_ids(self) -> np.ndarray:
        return self.obs["drug_id"].to_numpy()

    @property
    def size(self) -> int:
        return self.obs.shape[0]

    @property
    def n_cells(self) -> int:
        return self.obs["cell_id"].unique().size

    @property
    def n_drugs(self) -> int:
        return self.obs["drug_id"].unique().size

    def select(
        self,
        ids: list[str],
        # cell_ids: t.Iterable[str] | None = None,
        # drug_ids: t.Iterable[str] | None = None,
        **dataset_args,
    ) -> Dataset:
        """"""
        # FIXME: do we need deep copy here?
        # FIXME: do I want this to be a copy (probably)?
        # NOTE: we might only need the copy when both args are None
        cell_features = copy.deepcopy(self.cell_features)
        drug_features = copy.deepcopy(self.drug_features)
        obs = self.obs[self.obs["id"].isin(ids)].copy(deep=True)

        # FIXME: I should filter the index or raise a warning for
        #   missing cell ids
        cell_ids = obs["cell_id"].unique().tolist()
        drug_ids = obs["drug_id"].unique().tolist()

        cell_features = {k: v.loc[cell_ids] for k, v in cell_features.items()}
        drug_features = {k: v.loc[drug_ids] for k, v in drug_features.items()}

        return Dataset(obs, cell_features, drug_features, **dataset_args)

    def transform_labels(
        self, func: t.Callable[[pd.Series], pd.Series]
    ) -> Dataset:
        """"""
        self.obs["label"] = func(self.obs["label"])
        return self

    def transform_cell_feature(
        self, feature: str, func: t.Callable[[pd.DataFrame], pd.DataFrame]
    ) -> Dataset:
        """"""
        if feature not in self.cell_features:
            cell_feature_keys = tuple(self.cell_features.keys())
            raise KeyError(f"Feature must be one of {cell_feature_keys}")

        X = self.cell_features[feature]
        self.cell_features[feature] = func(X)

        return self

    def transform_drug_feature(
        self, feature: str, func: t.Callable[[pd.DataFrame], pd.DataFrame]
    ) -> Dataset:
        """"""
        if feature not in self.drug_features:
            drug_feature_keys = tuple(self.drug_features.keys())
            raise KeyError(f"feature must be one of {drug_feature_keys}")

        X = self.drug_features[feature]
        self.drug_features[feature] = func(X)

        return self

    def encode_batches(
        self,
        cell_features: list[str],
        drug_features: list[str],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> t.Generator[
        tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray], None, None
    ]:
        """"""
        obs = self.obs.copy(deep=True)
        if shuffle:
            obs = obs.sample(frac=1)

        split_inds = np.arange(batch_size, obs.shape[0], batch_size)
        for batch in np.array_split(obs, split_inds):
            batch_cell_ids = batch["cell_id"].values
            batch_drug_ids = batch["drug_id"].values
            batch_labels = batch["label"].values

            enc_cell_features = self._encode_cells(
                cell_features, batch_cell_ids
            )
            enc_drug_features = self._encode_drugs(
                drug_features, batch_drug_ids
            )
            batch_features = enc_cell_features + enc_drug_features

            yield (
                batch_features,
                batch_labels,
                batch_cell_ids,
                batch_drug_ids,
            )

    def encode(
        self, cell_features: list[str], drug_features: list[str]
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """"""
        cell_ids = self.cell_ids
        drug_ids = self.drug_ids
        enc_cell_features = self._encode_cells(cell_features, cell_ids)
        enc_drug_features = self._encode_drugs(drug_features, drug_ids)
        enc_features = enc_cell_features + enc_drug_features

        return (enc_features, self.labels, cell_ids, drug_ids)

    def _encode_cells(
        self, features: list[str], cell_ids: list[str]
    ) -> list[np.ndarray]:
        """"""
        enc_features = []
        for feature in features:
            if feature not in self.cell_features:
                cell_feature_keys = tuple(self.cell_features.keys())
                raise KeyError(f"features must be one of {cell_feature_keys}")
            enc_feature = self.cell_features[feature].loc[cell_ids]
            enc_features.append(enc_feature.to_numpy())

        return enc_features

    def _encode_drugs(
        self, features: list[str], drug_ids: list[str]
    ) -> list[np.ndarray]:
        """"""
        enc_features = []
        for feature in features:
            if feature not in self.drug_features:
                drug_feature_keys = tuple(self.drug_features.keys())
                raise KeyError(f"features must be one of {drug_feature_keys}")
            enc_feature = self.drug_features[feature].loc[drug_ids]
            enc_features.append(enc_feature.to_numpy())

        return enc_features

    def make_tf_dataset(
        self,
        cell_features: list[str],
        drug_features: list[str],
        include_metadata: bool = False,
    ) -> tf.data.Dataset:
        """"""
        # FIXME: remove cell_features and drug_features args
        #   the dataset should just use all available features
        X, y = self.encode(cell_features, drug_features)
        feature_ds = (tf.data.Dataset.from_tensor_slices(arr) for arr in X)
        feature_ds = tf.data.Dataset.zip(tuple(feature_ds))
        label_ds = tf.data.Dataset.from_tensor_slices(y)
        tf_ds = (feature_ds, label_ds)
        if include_metadata:
            cell_id_ds = tf.data.Dataset.from_tensor_slices(self.cell_ids)
            drug_id_ds = tf.data.Dataset.from_tensor_slices(self.drug_ids)
            tf_ds += (cell_id_ds, drug_id_ds)
        tf_ds = tf.data.Dataset.zip(tf_ds)

        return tf_ds
