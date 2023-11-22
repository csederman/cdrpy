"""
Dataset preprocessing utilities.
"""

from __future__ import annotations

import typing as t

from sklearn.preprocessing import StandardScaler

from ..feat.transformers import GroupStandardScaler


if t.TYPE_CHECKING:
    from .datasets import Dataset


NormMethod = t.Literal["global", "grouped"]


def normalize_responses(
    train_ds: Dataset,
    val_ds: Dataset | None = None,
    test_ds: Dataset | None = None,
    norm_method: NormMethod = "global",
) -> t.Tuple[Dataset, Dataset | None, Dataset | None]:
    """"""
    if norm_method == "global":
        # FIXME: redifine this as a PandasStandardScaler to avoid explicit "label"
        ss = StandardScaler()
        train_ds.obs["label"] = ss.fit_transform(train_ds.obs[["label"]])

        if val_ds is not None:
            val_ds.obs["label"] = ss.transform(val_ds.obs[["label"]])

        if test_ds is not None:
            test_ds.obs["label"] = ss.transform(test_ds.obs[["label"]])

    elif norm_method == "grouped":
        gss = GroupStandardScaler()
        train_ds.obs["label"] = gss.fit_transform(
            train_ds.obs[["label"]], groups=train_ds.obs["drug_id"]
        )

        if val_ds is not None:
            val_ds.obs["label"] = gss.transform(
                val_ds.obs["label"], groups=val_ds.obs["label"]
            )

        if test_ds is not None:
            test_ds.obs["label"] = gss.transform(
                test_ds.obs["label"], groups=test_ds.obs["label"]
            )

    else:
        norm_methods = ("global", "grouped")
        ValueError(f"norm_method must be one of {norm_methods}")

    return train_ds, val_ds, test_ds
