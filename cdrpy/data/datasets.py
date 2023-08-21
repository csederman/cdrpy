"""

"""

from __future__ import annotations

import yaml

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

from dataclasses import dataclass
from pathlib import Path
from tensorflow import keras

from cdrpy.util.decorators import check_encoders

if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import Encoder
    from cdrpy.types import PathLike


EncodedDataset = tuple[t.Any, np.ndarray]
EncodedDatasetWithIds = tuple[t.Any, np.ndarray, np.ndarray]


@dataclass(repr=False)
class Dataset:
    """Cancer drug response dataset."""

    obs: pd.DataFrame
    name: str | None = None
    desc: str | None = None
    cell_encoders: list[Encoder] | None = None
    drug_encoders: list[Encoder] | None = None
    encode_drugs_first: bool = False

    def __post_init__(self) -> None:
        required_cols = ["id", "cell_id", "drug_id", "label"]
        assert all(c in self.obs.columns for c in required_cols)

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name}, size={self.size:_}, "
            f"n_cells={self.n_cells:_}, n_drugs={self.n_drugs:_})"
        )

    @property
    @check_encoders
    def encoders(self) -> list[Encoder] | None:
        if self.encode_drugs_first:
            return self.drug_encoders + self.cell_encoders
        return self.cell_encoders + self.drug_encoders

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

    def register_encoders(
        self, cell_encoders: list[Encoder], drug_encoders: list[Encoder]
    ) -> None:
        """Registers encoders with the dataset."""
        # FIXME: add validation of encoders (check all ids in self.obs are in
        #   the encoder keys)
        self.cell_encoders = cell_encoders
        self.drug_encoders = drug_encoders

    @check_encoders
    def _generator(
        self, return_ids: bool = False, as_numpy: bool = False
    ) -> t.Generator[EncodedDataset | EncodedDatasetWithIds, None, None]:
        """"""
        for _, row in self.obs.iterrows():
            cell_id = row["cell_id"]
            drug_id = row["drug_id"]

            cell_feat = [e.get(cell_id) for e in self.cell_encoders]
            drug_feat = [e.get(drug_id) for e in self.drug_encoders]
            features = tuple(cell_feat + drug_feat)

            if return_ids:
                yield (features, row["label"], cell_id, drug_id)

            yield (features, row["label"])

    @check_encoders
    def encode(
        self, return_ids: bool = False, as_numpy: bool = False
    ) -> EncodedDataset | EncodedDatasetWithIds:
        """"""
        # FIXME: add check that encoders return arrays
        # FIXME: add interface for encoder type (has encoder.encode method)
        return encode(
            self.obs,
            self.cell_encoders,
            self.drug_encoders,
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
        split_inds = np.arange(batch_size, self.size, batch_size)
        for batch_df in np.array_split(self.obs, split_inds):
            yield encode(
                batch_df,
                self.cell_encoders,
                self.drug_encoders,
                return_ids=return_ids,
                as_numpy=as_numpy,
            )

    @check_encoders
    def encode_tf_legacy(self, return_ids: bool = False) -> tf.data.Dataset:
        """"""
        return encode_tf(
            self.obs,
            self.cell_encoders,
            self.drug_encoders,
            return_ids,
            drugs_first=self.encode_drugs_first,
        )

    @check_encoders
    def encode_tf(self, return_ids: bool = False) -> tf.data.Dataset:
        """"""
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=self._infer_tf_output_signature(),
        )

    def _infer_tf_output_signature(self) -> t.Any:
        # Add a drugs first option
        encoders: list[Encoder] = self.encoders
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
        obs = self.obs[self.obs["id"].isin(ids)].copy(deep=True)
        return Dataset(
            obs,
            cell_encoders=self.cell_encoders,
            drug_encoders=self.drug_encoders,
            encode_drugs_first=self.encode_drugs_first,
            **kwargs,
        )

    def sample(self, n: int, **kwargs) -> Dataset:
        """Samples a random subset of `n` drug response observations."""
        return self.select(self.obs["id"].sample(n=n), **kwargs)

    def shuffle(self, random_state: t.Any = None) -> None:
        """Shuffles the drug response observations."""
        self.obs = self.obs.sample(frac=1, random_state=random_state)

    def get_config(self) -> dict:
        """"""
        config = {
            "name": self.name,
            "desc": self.desc,
            "class": self.__class__.__name__,
        }
        config["cell_encoders"] = [e.get_config() for e in self.cell_encoders]
        config["drug_encoders"] = [e.get_config() for e in self.drug_encoders]
        return config

    @classmethod
    def from_csv(cls, file_path: str, **kwargs) -> Dataset:
        """"""
        df = pd.read_csv(file_path, dtype={"label": np.float32})
        return cls(df, **kwargs)

    def save(self, dir_: PathLike | Path) -> None:
        """Saves the dataset configuration and data."""
        dir_ = Path(dir_)  # FIXME: this might not work for some pathlikes
        ds_conf = self.get_config()
        self.obs.to_pickle(dir_ / "obs.pickle")

        items = zip(self.cell_encoders, ds_conf["cell_encoders"])
        for i, (enc, enc_conf) in enumerate(items, 1):
            enc_conf["source"] = f"cell_{i}.pickle"
            enc.to_pickle(dir_ / enc_conf["source"])

        items = zip(self.drug_encoders, ds_conf["drug_encoders"])
        for i, (enc, enc_conf) in enumerate(items, 1):
            enc_conf["source"] = f"drug_{i}.pickle"
            enc.to_pickle(dir_ / enc_conf["source"])

        with open(dir_ / "config.yaml", "w") as fh:
            yaml.dump(ds_conf, fh, sort_keys=False)

    @classmethod
    def load(cls, dir_: PathLike | Path) -> Dataset:
        """Loads a saved dataset."""
        import importlib

        module = importlib.import_module("cdrpy.feat.encoders")

        dir_ = Path(dir_)  # FIXME: may not work for all pathlikes

        obs = pd.read_pickle(dir_ / "obs.pickle")
        with open(dir_ / "config.yaml", "rb") as fh:
            ds_conf = yaml.safe_load(fh)

        cell_encoders = []
        for enc_conf in ds_conf["cell_encoders"]:
            class_ = getattr(module, enc_conf["class"])
            enc = class_.from_pickle(
                dir_ / enc_conf["source"], name=enc_conf["name"]
            )
            cell_encoders.append(enc)

        drug_encoders = []
        for enc_conf in ds_conf["drug_encoders"]:
            class_ = getattr(module, enc_conf["class"])
            enc = class_.from_pickle(
                dir_ / enc_conf["source"], name=enc_conf["name"]
            )
            drug_encoders.append(enc)

        return cls(
            obs,
            name=ds_conf["name"],
            desc=ds_conf["desc"],
            cell_encoders=cell_encoders,
            drug_encoders=drug_encoders,
        )


def _extract_column_values(df: pd.DataFrame) -> tuple[np.ndarray, 3]:
    """Extract column values."""
    cell_ids = df["cell_id"].values
    drug_ids = df["drug_id"].values
    labels = df["label"].values
    return (cell_ids, drug_ids, labels)


def encode(
    df: pd.DataFrame,
    cell_encoders: list[Encoder],
    drug_encoders: list[Encoder],
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
    cell_encoders: list[Encoder],
    drug_encoders: list[Encoder],
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
        batch_gen = ds.encode_batches(
            batch_size, return_ids=True, as_numpy=True
        )
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
