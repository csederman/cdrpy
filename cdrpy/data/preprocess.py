"""
Dataset preprocessing utilities.
"""

from __future__ import annotations

import typing as t

from sklearn.preprocessing import StandardScaler

from ..feat.transformers import PandasGroupedStandardScaler


if t.TYPE_CHECKING:
    from .datasets import Dataset


NormMethod = t.Literal["global", "grouped"]


def normalize_responses(
    train_ds: Dataset,
    val_ds: Dataset | None = None,
    test_ds: Dataset | None = None,
    norm_method: NormMethod = "global",
    value_col: str = "label",
    group_col: str = "drug_id",
) -> t.Tuple[Dataset, Dataset | None, Dataset | None]:
    """"""
    if norm_method == "global":
        # FIXME: redifine this as a PandasStandardScaler to avoid explicit "label"
        scaler = StandardScaler()
        train_ds.obs[value_col] = scaler.fit_transform(train_ds.obs[[value_col]])
        if val_ds is not None:
            val_ds.obs[value_col] = scaler.transform(val_ds.obs[[value_col]])
        if test_ds is not None:
            test_ds.obs[value_col] = scaler.transform(test_ds.obs[[value_col]])
    elif norm_method == "grouped":
        scaler = PandasGroupedStandardScaler(value_col=value_col, group_col=group_col)
        train_ds.obs = scaler.fit_transform(train_ds.obs)
        if val_ds is not None:
            val_ds.obs = scaler.transform(val_ds.obs)
        if test_ds is not None:
            test_ds.obs = scaler.transform(test_ds.obs)
    else:
        norm_methods = ("global", "grouped")
        ValueError(f"norm_method must be one of {norm_methods}")

    return train_ds, val_ds, test_ds
