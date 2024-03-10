""""""

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

from cdrpy.util.decorators import check_encoders, deprecated
from cdrpy.feat.encoders import Encoder, EncoderMapper
from cdrpy.util import io
from cdrpy.datasets.utils import find_cdrpy_dataset_dir

if t.TYPE_CHECKING:
    from cdrpy.transforms import Transform
    from collections import Sequence


Self = t.TypeVar("Self", bound="BaseDataset")


EncoderDict = t.Dict[str, Encoder]
EncodedDataset = t.Tuple[t.Any, np.ndarray]
EncodedDatasetWithIds = t.Tuple[t.Any, np.ndarray, np.ndarray]


class BaseDataset(ABC):
    """Base class for cancer drug response dataset.

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

    _required_obs_cols: t.List[str] = []

    def __init__(
        self,
        obs: pd.DataFrame,
        cell_encoders: EncoderDict | None = None,
        drug_encoders: EncoderDict | None = None,
        cell_meta: pd.DataFrame | None = None,
        drug_meta: pd.DataFrame | None = None,
        transforms: Sequence[Transform] | None = None,
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

        if transforms is not None:
            if not isinstance(transforms, (list, tuple)) and callable(transforms):
                transforms = [transforms]
            elif not all(callable(t) for t in transforms):
                raise ValueError(
                    "`transforms` must be callable or a list of callables."
                )
            for transform in transforms:
                self.apply(transform)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, size={self.size:_}, "
            f"n_cells={self.n_cells:_}, n_drugs={self.n_drugs:_})"
        )

    def _validate_obs(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Checks that the required columns are in the observations DataFrame."""
        if not all(c in obs.columns for c in self._required_obs_cols):
            raise ValueError("Missing required columns")
        return obs

    def apply(self, transform: Transform | None) -> None:
        """Applies a transformation to the dataset."""
        if not callable(transform):
            raise ValueError("`transform` must be callable")

        transform(self)

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
        return tf.as_dtype(self.obs["label"].dtype)

    @property
    def labels(self) -> np.ndarray:
        return self.obs["label"].values

    @property
    def cell_ids(self) -> np.ndarray:
        return self.obs["cell_id"].values

    @property
    def size(self) -> int:
        return self.obs.shape[0]

    @property
    def n_cells(self) -> int:
        return self.obs["cell_id"].unique().size

    @property
    @abstractmethod
    def n_drugs(self) -> int:
        pass

    def select(self: Self, ids: t.Iterable[str], **kwargs) -> Self:
        """"""
        # FIXME: do we need this deep copy?
        # FIXME: add optional param to select and subset from the encoders
        selected_obs = self.obs[self.obs["id"].isin(ids)].copy(deep=True)

        return type(self)(
            selected_obs,
            cell_encoders=self.cell_encoders,
            drug_encoders=self.drug_encoders,
            cell_meta=self.cell_meta,
            drug_meta=self.drug_meta,
            encode_drugs_first=self.encode_drugs_first,
            **kwargs,
        )

    @abstractmethod
    def select_cells(self: Self, cell_ids: t.Iterable[str], **kwargs) -> Self:
        pass

    @abstractmethod
    def select_drugs(self: Self, drug_ids: t.Iterable[str], **kwargs) -> Self:
        pass

    def sample(self: Self, n: int, seed: t.Any = None, **kwargs) -> Self:
        """Samples a random subset of `n` drug response observations."""
        return self.select(self.obs["id"].sample(n=n, random_state=seed), **kwargs)

    @abstractmethod
    def sample_cells(self: Self, n: int, **kwargs) -> Self:
        pass

    @abstractmethod
    def sample_drugs(self: Self, n: int, **kwargs) -> Self:
        pass

    def shuffle(self, seed: t.Any = None) -> None:
        """Shuffles the drug response observations."""
        self.obs = self.obs.sample(frac=1, random_state=seed)

    @classmethod
    def from_csv(cls: type[Self], file_path: str, **kwargs) -> Self:
        """"""
        obs = pd.read_csv(file_path, dtype={"label": np.float32})
        return cls(obs, **kwargs)


class Dataset(BaseDataset):
    """Monotherapy dataset."""

    _required_obs_cols = ["id", "cell_id", "drug_id", "label"]

    @property
    def drug_ids(self) -> np.ndarray:
        return self.obs["drug_id"].values

    @property
    def n_drugs(self) -> int:
        return self.obs["drug_id"].unique().size

    def select_cells(self, cell_ids: t.Iterable[str], **kwargs) -> Dataset:
        """Creates a subset with the specified cell ids."""
        obs_ids = self.obs[self.obs["cell_id"].isin(cell_ids)]["id"]
        return self.select(obs_ids, **kwargs)

    def select_drugs(self, drug_ids: t.Iterable[str], **kwargs) -> Dataset:
        """Creates a subset with the specified drug ids."""
        obs_ids = self.obs[self.obs["drug_id"].isin(drug_ids)]["id"]
        return self.select(obs_ids, **kwargs)

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

    @deprecated(
        message="Dataset.save will be deprecated in favor of a more flexible format."
    )
    def save(self: Dataset, file_path: str | Path) -> None:
        """Saves the dataset to hdf5 format."""

        # FIXME: this will be deprecated in favor of a more flexible storage
        #   format that works more efficiently for large datasets with
        #   variable feature formats

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
    @deprecated(
        message="Dataset.load will be deprecated in favor of a more flexible format."
    )
    def load(cls, file_path: str | Path) -> Dataset:
        """Loads a stored dataset from an hdf5 file."""

        # FIXME: this will be deprecated in favor of a more flexible storage
        #   format that works more efficiently for large datasets with
        #   variable feature formats

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


class CombinationDataset(BaseDataset):

    _required_obs_cols = ["id", "cell_id", "drug_1_id", "drug_2_id", "label"]

    @property
    def drug_1_ids(self) -> np.ndarray:
        return self.obs["drug_1_id"].values

    @property
    def drug_2_ids(self) -> np.ndarray:
        return self.obs["drug_2_id"].values

    @property
    def n_drugs(self) -> int:
        return len(set(self.drug_1_ids) | set(self.drug_2_ids))

    def select_cells(self, cell_ids: t.Iterable[str], **kwargs) -> CombinationDataset:
        """Creates a subset with the specified cell ids."""
        obs_ids = self.obs[self.obs["cell_id"].isin(cell_ids)]["id"]
        return self.select(obs_ids, **kwargs)

    def select_drugs(self, drug_ids: t.Iterable[str], **kwargs) -> CombinationDataset:
        """Creates a subset with the specified drug ids."""
        obs_ids = self.obs[
            self.obs["drug_1_id"].isin(drug_ids) | self.obs["drug_2_id"].isin(drug_ids)
        ]["id"]
        return self.select(obs_ids, **kwargs)

    def sample_cells(self, n: int, **kwargs) -> CombinationDataset:
        """Samples a random subset of `n` cells and their drug responses."""
        choices = list(set(self.cell_ids))
        sampled_cells = random.sample(choices, n)
        return self.select_cells(sampled_cells, **kwargs)

    def sample_drugs(self, n: int, **kwargs) -> CombinationDataset:
        """Samples a random subset of `n` drugs and their responses."""
        choices = list(set(self.drug_1_ids) | set(self.drug_2_ids))
        sampled_drugs = random.sample(choices, n)
        return self.select_drugs(sampled_drugs, **kwargs)


class CustomDataset(Dataset, ABC):
    """"""

    name = None
    desc = None
    url = None

    def __init__(self, **kwargs) -> None:
        if not os.path.exists(self.path):
            self.download()

        obs, c_enc, d_enc, c_meta, d_meta = self.read()

        # FIXME: decide how to handle the `name` and `desc` args
        for kwarg in ("name", "desc"):
            if kwarg in kwargs:
                kwargs.pop(kwarg)

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

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def read(
        self,
    ) -> t.Tuple[pd.DataFrame, EncoderDict, EncoderDict, pd.DataFrame, pd.DataFrame]:
        pass

    @property
    def path(self) -> str:
        if self.name is None:
            raise ValueError("fixme")
        return os.path.join(find_cdrpy_dataset_dir(), self.name)

    def joinpath(self, path: str) -> str:
        return os.path.join(self.path, path)
