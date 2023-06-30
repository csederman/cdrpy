"""

"""

from __future__ import annotations

import os
import copy

import numpy as np
import pandas as pd
import typing as t

from dataclasses import dataclass
from pathlib import Path


PathLike = t.Union[str, bytes, os.PathLike]
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
        cell_ids: t.Iterable[str] | None = None,
        drug_ids: t.Iterable[str] | None = None,
        **dataset_args,
    ) -> Dataset:
        """"""
        # FIXME: do we need deep copy here?
        # FIXME: do I want this to be a copy (probably)?
        # NOTE: we might only need the copy when both args are None
        obs = self.obs.copy(deep=True)
        cell_features = copy.deepcopy(self.cell_features)
        drug_features = copy.deepcopy(self.drug_features)

        if cell_ids is not None:
            obs = obs[obs["cell_id"].isin(cell_ids)]
            cell_features = {
                # FIXME: I should filter the index or raise a warning for
                #   missing cell ids
                k: v.loc[cell_ids]
                for k, v in cell_features.items()
            }

        if drug_ids is not None:
            obs = obs[obs["drug_id"].isin(drug_ids)]
            drug_features = {
                # FIXME: I should filter the index or raise a warning for
                #   missing drug ids
                k: v.loc[drug_ids]
                for k, v in drug_features.items()
            }

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

    def encode(
        self, cell_features: list[str], drug_features: list[str]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """"""
        enc_cell_features = self._encode_cell_features(cell_features)
        enc_drug_features = self._encode_drug_features(drug_features)
        enc_features = enc_cell_features + enc_drug_features

        return (enc_features, self.labels)

    def _encode_cell_features(self, features: list[str]) -> list[np.ndarray]:
        """"""
        enc_features = []
        for feature in features:
            if feature not in self.cell_features:
                cell_feature_keys = tuple(self.cell_features.keys())
                raise KeyError(f"features must be one of {cell_feature_keys}")
            enc_feature = self.cell_features[feature].loc[self.cell_ids]
            enc_features.append(enc_feature.to_numpy())

        return enc_features

    def _encode_drug_features(self, features: list[str]) -> list[np.ndarray]:
        """"""
        enc_features = []
        for feature in features:
            if feature not in self.drug_features:
                drug_feature_keys = tuple(self.drug_features.keys())
                raise KeyError(f"features must be one of {drug_feature_keys}")
            enc_feature = self.drug_features[feature].loc[self.drug_ids]
            enc_features.append(enc_feature.to_numpy())

        return enc_features
