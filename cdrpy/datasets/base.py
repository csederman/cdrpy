"""

"""

from __future__ import annotations

import os
import h5py
import random

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

from abc import ABC, abstractmethod
from pathlib import Path
from tensorflow import keras

from cdrpy.util.decorators import check_encoders
from cdrpy.feat.encoders import Encoder, EncoderMapper
from cdrpy.util import io
from cdrpy.data.utils.store import find_cdrpy_dataset_dir

if t.TYPE_CHECKING:
    from cdrpy.types import PathLike


EncoderDict = t.Dict[str, Encoder]
EncodedDataset = t.Tuple[t.Any, np.ndarray]
EncodedDatasetWithIds = t.Tuple[t.Any, np.ndarray, np.ndarray]


class Dataset:
    """Cancer drug response dataset.

    Parameters
    ----------
        obs:
        cell_encoders:
        drug_encoders:
        cell_meta:
        drug_meta:
        encode_drugs_first:
        name:
        desc:
        seed:
    """

    def __init__(
        self,
        obs: pd.DataFrame,
        cell_encoders: EncoderDict | None = None,
        drug_encoders: EncoderDict | None = None,
        cell_meta: pd.DataFrame | None = None,
        drug_meta: pd.DataFrame | None = None,
        encode_drugs_first: bool = False,
        name: str | None = None,
        desc: str | None = None,
    ) -> None:
        self.obs = self._validate_obs(obs)
        self.cell_encoders = cell_encoders
        self.drug_encoders = drug_encoders
        self.cell_meta = cell_meta
        self.drug_meta = drug_meta
        self.encode_drugs_first = encode_drugs_first
        self.name = name
        self.desc = desc

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name}, size={self.size:_}, "
            f"n_cells={self.n_cells:_}, n_drugs={self.n_drugs:_})"
        )

    @staticmethod
    def _validate_obs(obs: pd.DataFrame) -> pd.DataFrame:
        """Checks that the required columns are in the observations."""
        required_cols = ["id", "cell_id", "drug_id", "label"]
        if not all(c in obs.columns for c in required_cols):
            raise ValueError("Missing required columns")
        return obs[required_cols]

    @property
    @check_encoders
    def encoders(self) -> t.List[Encoder] | None:
        cell_encoders = list(self.cell_encoders.values())
        drug_encoders = list(self.drug_encoders.values())
        if self.encode_drugs_first:
            return drug_encoders + cell_encoders
        return cell_encoders + drug_encoders

    @property
    def dtype(self) -> tf.dtypes.DType:
        return tf.as_dtype(self.labels["label"].dtype)

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

    @check_encoders
    def _generator(
        self, return_ids: bool = False, as_numpy: bool = False
    ) -> t.Generator[EncodedDataset | EncodedDatasetWithIds, None, None]:
        """"""
        # FIXME: deprecated
        for _, row in self.obs.iterrows():
            cell_id = row["cell_id"]
            drug_id = row["drug_id"]

            cell_encoders = list(self.cell_encoders.values())
            drug_encoders = list(self.drug_encoders.values())

            cell_feat = [e.get(cell_id) for e in cell_encoders]
            drug_feat = [e.get(drug_id) for e in drug_encoders]

            features = tuple(cell_feat + drug_feat)

            if return_ids:
                yield (features, row["label"], cell_id, drug_id)

            yield (features, row["label"])

    @check_encoders
    def encode(
        self, return_ids: bool = False, as_numpy: bool = False
    ) -> EncodedDataset | EncodedDatasetWithIds:
        """"""
        # FIXME: make this take the cell_ids and drug_ids to encoder and change
        #   to encode_features
        # FIXME: add check that encoders return arrays
        # FIXME: add interface for encoder type (has encoder.encode method)
        cell_encoders = list(self.cell_encoders.values())
        drug_encoders = list(self.drug_encoders.values())

        return encode(
            self.obs,
            cell_encoders,
            drug_encoders,
            return_ids=return_ids,
            as_numpy=as_numpy,
            drugs_first=self.encode_drugs_first,
        )

    @check_encoders
    def encode_batches(
        self,
        batch_size: int = 32,
        return_ids: bool = False,
        as_numpy: bool = False,
    ) -> t.Generator[EncodedDataset | EncodedDatasetWithIds, None, None]:
        """"""
        # FIXME: deprecated
        cell_encoders = list(self.cell_encoders.values())
        drug_encoders = list(self.drug_encoders.values())

        split_inds = np.arange(batch_size, self.size, batch_size)
        for batch_df in np.array_split(self.obs, split_inds):
            yield encode(
                batch_df,
                cell_encoders,
                drug_encoders,
                return_ids=return_ids,
                as_numpy=as_numpy,
            )

    @check_encoders
    def encode_tf(self, return_ids: bool = False) -> tf.data.Dataset:
        """"""
        # FIXME: deprecated
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=self._infer_tf_output_signature(),
        )

    def _infer_tf_output_signature(self) -> t.Any:
        # FIXME: make this a method of the encoder or a funciton that wraps
        #   the encoder
        # Add a drugs first option
        encoders: t.List[Encoder] = self.encoders
        get_tensor_spec = lambda enc: tf.TensorSpec(
            enc.shape, tf.as_dtype(enc.dtype), name=enc.name
        )
        return (
            tuple(get_tensor_spec(enc) for enc in encoders),
            tf.TensorSpec(shape=(), dtype=tf.float32, name="label"),
        )

    def select(self, ids: t.Iterable[str], **kwargs) -> Dataset:
        """"""
        # FIXME: do we need this deep copy?
        # FIXME: add optional param to select and subset from the encoders
        obs = self.obs[self.obs["id"].isin(ids)].copy(deep=True)

        return Dataset(
            obs,
            cell_encoders=self.cell_encoders,
            drug_encoders=self.drug_encoders,
            cell_meta=self.cell_meta,
            drug_meta=self.drug_meta,
            encode_drugs_first=self.encode_drugs_first,
            **kwargs,
        )

    def select_cells(self, cell_ids: t.Iterable[str], **kwargs) -> Dataset:
        """Creates a subset with the specified cell ids."""
        obs_ids = self.obs[self.obs["cell_id"].isin(cell_ids)]["id"]
        return self.select(obs_ids, **kwargs)

    def select_drugs(self, drug_ids: t.Iterable[str], **kwargs) -> Dataset:
        """Creates a subset with the specified drug ids."""
        obs_ids = self.obs[self.obs["drug_id"].isin(drug_ids)]["id"]
        return self.select(obs_ids, **kwargs)

    def sample(self, n: int, seed: t.Any = None, **kwargs) -> Dataset:
        """Samples a random subset of `n` drug response observations."""
        return self.select(self.obs["id"].sample(n=n, random_state=seed), **kwargs)

    def sample_cells(self, n: int, **kwargs) -> Dataset:
        """Samples a random subset of `n` cells and their drug responses."""
        choices = list(set(self.cell_ids))
        sampled_cells = random.sample(choices, n)
        return self.select_cells(sampled_cells, **kwargs)

    def sample_drugs(self, n: int, **kwargs) -> Dataset:
        """Samples a random subset of `n` drugs and their responses."""
        choices = list(set(self.drug_ids))
        sampled_drugs = random.sample(choices, n)
        return self.select_drugs(sampled_drugs, **kwargs)

    def shuffle(self, seed: t.Any = None) -> None:
        """Shuffles the drug response observations."""
        self.obs = self.obs.sample(frac=1, random_state=seed)

    @classmethod
    def from_csv(cls, file_path: str, **kwargs) -> Dataset:
        """"""
        df = pd.read_csv(file_path, dtype={"label": np.float32})
        return cls(df, **kwargs)

    def save(self: Dataset, file_path: str | Path) -> None:
        """Saves the dataset to hdf5 format."""

        # TODO: add saving fo metadata containers

        with h5py.File(file_path, "w") as f:
            # store options
            f.attrs["encode_drugs_first"] = int(self.encode_drugs_first)
            if self.name is not None:
                f.attrs["name"] = self.name
            if self.desc is not None:
                f.attrs["desc"] = self.desc

            # save the drug responses
            group = f.create_group("obs")
            io.pandas_to_h5(group, self.obs)

            # save the cell_encoders
            if self.cell_encoders is not None:
                f.attrs["has_cell_encoders"] = 1
                group = f.create_group("cell_encoders")
                for key, enc in self.cell_encoders.items():
                    enc.save(group, key)
                    group.attrs[key] = enc.__class__.__name__
            else:
                f.attrs["has_cell_encoders"] = 0

            # save the drug encoders
            if self.drug_encoders is not None:
                f.attrs["has_drug_encoders"] = 1
                group = f.create_group("drug_encoders")
                for key, enc in self.drug_encoders.items():
                    enc.save(group, key)
                    group.attrs[key] = enc.__class__.__name__
            else:
                f.attrs["has_drug_encoders"] = 0

            # save cell metadata
            if isinstance(self.cell_meta, pd.DataFrame):
                group = f.create_group("cell_meta")
                io.pandas_to_h5(group, self.cell_meta, index=True)

            # save drug metadata
            if isinstance(self.drug_meta, pd.DataFrame):
                group = f.create_group("drug_meta")
                io.pandas_to_h5(group, self.drug_meta, index=True)

    @classmethod
    def load(cls, file_path: str | Path) -> Dataset:
        """Loads a stored dataset from an hdf5 file."""

        # TODO: add loading of metadata containers

        with h5py.File(file_path, "r") as f:
            name = f.attrs.get("name")
            desc = f.attrs.get("desc")
            encode_drugs_first = bool(f.attrs["encode_drugs_first"])

            # load the drug response observations
            obs = io.pandas_from_h5(f["obs"])

            # load the cell encoders
            cell_encoders = None
            if f.attrs["has_cell_encoders"] == 1:
                group = f["cell_encoders"]
                cell_encoders = {}
                for key in group.keys():
                    encoder_cls = EncoderMapper[group.attrs[key]]
                    cell_encoders[key] = encoder_cls.load(group, key)

            # load the drug encoders
            drug_encoders = None
            if f.attrs["has_drug_encoders"] == 1:
                group = f["drug_encoders"]
                drug_encoders = {}
                for key in group.keys():
                    encoder_cls = EncoderMapper[group.attrs[key]]
                    drug_encoders[key] = encoder_cls.load(group, key)

            # load the cell metadata
            cell_meta = None
            cell_meta_group = f.get("cell_meta")
            if cell_meta_group is not None:
                cell_meta = io.pandas_from_h5(cell_meta_group)

            # load the drug metadata
            drug_meta = None
            drug_meta_group = f.get("drug_meta")
            if drug_meta_group is not None:
                drug_meta = io.pandas_from_h5(drug_meta_group)

        return cls(
            obs,
            cell_encoders,
            drug_encoders,
            cell_meta=cell_meta,
            drug_meta=drug_meta,
            encode_drugs_first=encode_drugs_first,
            name=name,
            desc=desc,
        )


def _extract_column_values(df: pd.DataFrame) -> t.Tuple[np.ndarray, 3]:
    """Extract column values."""
    cell_ids = df["cell_id"].values
    drug_ids = df["drug_id"].values
    labels = df["label"].values
    return (cell_ids, drug_ids, labels)


def encode(
    df: pd.DataFrame,
    cell_encoders: t.List[Encoder],
    drug_encoders: t.List[Encoder],
    return_ids: bool = False,
    as_numpy: bool = False,
    drugs_first: bool = False,
) -> EncodedDataset | EncodedDatasetWithIds:
    """"""
    cell_ids, drug_ids, labels = _extract_column_values(df)

    cell_feat = tuple(e.encode(cell_ids) for e in cell_encoders)
    drug_feat = tuple(e.encode(drug_ids) for e in drug_encoders)

    if as_numpy:
        cell_feat = tuple(np.array(f) for f in cell_feat)
        drug_feat = tuple(np.array(f) for f in drug_feat)

    if drugs_first:
        features = drug_feat + cell_feat
    else:
        features = cell_feat + drug_feat

    if return_ids:
        return (features, labels, cell_ids, drug_ids)
    return (features, labels)


def encode_tf(
    df: pd.DataFrame,
    cell_encoders: t.List[Encoder],
    drug_encoders: t.List[Encoder],
    return_ids: bool = False,
    drugs_first: bool = False,
) -> tf.data.Dataset:
    """"""
    cell_ids, drug_ids, labels = _extract_column_values(df)

    cell_features = [e.encode_tf(cell_ids) for e in cell_encoders]
    drug_features = [e.encode_tf(drug_ids) for e in drug_encoders]

    if drugs_first:
        features = tf.data.Dataset.zip((*drug_features, *cell_features))
    else:
        features = tf.data.Dataset.zip((*cell_features, *drug_features))

    labels = tf.data.Dataset.from_tensor_slices(labels)

    return tf.data.Dataset.zip((features, labels))


# FIXME: deprecated
def get_predictions(
    datasets: t.Iterable[Dataset],
    model: keras.Model,
    **kwargs,
) -> pd.DataFrame:
    """"""
    pred_df = []
    for ds in datasets:
        preds = model.predict(ds.encode_tf().batch(32)).reshape(-1)
        preds = pd.DataFrame(
            {
                "cell_id": ds.cell_ids,
                "drug_id": ds.drug_ids,
                "y_true": ds.labels,
                "y_pred": preds,
                "split": ds.name,
            }
        )
        pred_df.append(preds)

    pred_df = pd.concat(pred_df)
    for column, constant in kwargs.items():
        pred_df[column] = constant

    return pred_df


# FIXME: deprecated
def get_predictions_batches(
    datasets: t.Iterable[Dataset],
    model: keras.Model,
    batch_size: int,
    **kwargs,
) -> pd.DataFrame:
    """"""
    pred_df = []
    for ds in datasets:
        batch_df = []
        batch_gen = ds.encode_batches(batch_size, return_ids=True, as_numpy=True)
        for batch in batch_gen:
            batch_x, batch_y, batch_cells, batch_drugs = batch
            batch_preds = model.predict_on_batch(batch_x).reshape(-1)
            batch_df.append(
                pd.DataFrame(
                    {
                        "cell_id": batch_cells,
                        "drug_id": batch_drugs,
                        "y_true": batch_y,
                        "y_pred": batch_preds,
                    }
                )
            )
        batch_df = pd.concat(batch_df)
        batch_df["split"] = ds.name
        pred_df.append(batch_df)

    pred_df = pd.concat(pred_df)
    for column, constant in kwargs.items():
        pred_df[column] = constant

    return pred_df


class CustomDataset(Dataset, ABC):
    """"""

    name = None
    desc = None

    def __init__(self, **kwargs) -> None:
        if not os.path.exists(self.path):
            self.download()

        obs, c_enc, d_enc, c_meta, d_meta = self.read()

        # FIXME: decide how to handle the `name` and `desc` args
        if "name" in kwargs:
            kwargs.pop("name")

        if "desc" in kwargs:
            kwargs.pop("desc")

        super().__init__(
            obs,
            cell_encoders=c_enc,
            drug_encoders=d_enc,
            cell_meta=c_meta,
            drug_meta=d_meta,
            name=self.name,
            desc=self.desc,
            **kwargs,
        )

    @property
    def path(self) -> str:
        return os.path.join(find_cdrpy_dataset_dir(), self.name)

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def read(
        self,
    ) -> t.Tuple[pd.DataFrame, EncoderDict, EncoderDict, pd.DataFrame, pd.DataFrame]:
        pass
