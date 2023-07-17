"""

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

from dataclasses import dataclass


EncodedDataset = t.Union[
    t.Tuple[t.Any, np.ndarray], t.Tuple[t.Any, np.ndarray, np.ndarray]
]


@dataclass(repr=False)
class Dataset:
    """Cancer drug response dataset."""

    obs: pd.DataFrame
    name: str | None = None
    desc: str | None = None

    def __post_init__(self) -> None:
        required_cols = ["id", "cell_id", "drug_id", "label"]
        assert all(c in self.obs.columns for c in required_cols)

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name}, size={self.size:_}, "
            f"n_cells={self.n_cells:_}, n_drugs={self.n_drugs:_})"
        )

    @property
    def labels(self) -> np.ndarray:
        return self.obs["label"].values

    @property
    def cell_ids(self) -> np.ndarray:
        return self.obs["cell_id"].values

    @property
    def drug_ids(self) -> np.ndarray:
        return self.obs["drug_id"].values

    @property
    def size(self) -> int:
        return self.obs.shape[0]

    @property
    def n_cells(self) -> int:
        return self.obs["cell_id"].unique().size

    @property
    def n_drugs(self) -> int:
        return self.obs["drug_id"].unique().size

    def encode(
        self,
        cell_encoders: t.Sequence[t.Any],
        drug_encoders: t.Sequence[t.Any],
        return_ids: bool = False,
    ) -> EncodedDataset:
        """"""
        # FIXME: add check that encoders return arrays
        # FIXME: add interface for encoder type (has encoder.encode method)
        return encode(self, cell_encoders, drug_encoders, return_ids)

    def encode_tf(
        self,
        cell_encoders: t.Sequence[t.Any],
        drug_encoders: t.Sequence[t.Any],
        return_ids: bool = False,
    ) -> tf.data.Dataset:
        """"""
        return encode_tf(self, cell_encoders, drug_encoders, return_ids)

    def encode_batches(
        self,
        cell_encoders: t.Iterable[t.Any],
        drug_encoders: t.Iterable[t.Any],
        batch_size: int = 32,
        return_ids: bool = False,
    ) -> t.Generator[EncodedDataset, None, None]:
        """"""
        split_inds = np.arange(batch_size, self.size, batch_size)
        for batch_df in np.array_split(self.obs, split_inds):
            yield encode_df(batch_df, cell_encoders, drug_encoders, return_ids)

    def select(self, ids: t.Iterable[str], **kwargs) -> Dataset:
        """"""
        # FIXME: do we need this deep copy?
        obs = self.obs[self.obs["id"].isin(ids)].copy(deep=True)
        return Dataset(obs, **kwargs)

    def shuffle(self, random_state: t.Any = None) -> None:
        """Shuffle the drug response observations."""
        self.obs = self.obs.sample(frac=1, random_state=random_state)

    @classmethod
    def from_csv(cls, file_path: str, **kwargs) -> Dataset:
        """"""
        df = pd.read_csv(file_path, dtype={"label": np.float32})
        return cls(df, **kwargs)


def encode(
    ds: Dataset,
    cell_encoders: t.Sequence[t.Any],
    drug_encoders: t.Sequence[t.Any],
    return_ids: bool = False,
) -> EncodedDataset:
    """"""
    cell_feat = tuple(e.encode(ds.cell_ids) for e in cell_encoders)
    drug_feat = tuple(e.encode(ds.drug_ids) for e in drug_encoders)
    features = cell_feat + drug_feat

    if return_ids:
        return (features, ds.labels, ds.cell_ids, ds.drug_ids)
    return (features, ds.labels)


def encode_df(
    df: pd.DataFrame,
    cell_encoders: t.Sequence[t.Any],
    drug_encoders: t.Sequence[t.Any],
    return_ids: bool = False,
) -> EncodedDataset:
    """"""
    cell_ids = df["cell_id"].values
    drug_ids = df["drug_id"].values
    labels = df["label"].values

    cell_feat = tuple(e.encode(cell_ids) for e in cell_encoders)
    drug_feat = tuple(e.encode(drug_ids) for e in drug_encoders)
    features = cell_feat + drug_feat

    if return_ids:
        return (features, labels, cell_ids, drug_ids)
    return (features, labels)


def encode_tf(
    ds: Dataset,
    cell_encoders: t.Sequence[t.Any],
    drug_encoders: t.Sequence[t.Any],
    return_ids: bool = False,
) -> tf.data.Dataset:
    """"""
    cell_features = [e.encode_tf(ds.cell_ids) for e in cell_encoders]
    drug_features = [e.encode_tf(ds.drug_ids) for e in drug_encoders]

    features = tf.data.Dataset.zip((*cell_features, *drug_features))
    labels = tf.data.Dataset.from_tensor_slices(ds.labels)

    return tf.data.Dataset.zip((features, labels))
