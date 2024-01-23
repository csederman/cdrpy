""""""

from __future__ import annotations

import pandas as pd
import typing as t


def column_as_index(
    df: pd.DataFrame, index_col: str, index_name: str | None = None
) -> pd.DataFrame:
    if index_name is None:
        index_name = index_col
    return df.set_index(index_col).rename_axis(index=index_name)


def dict_from_columns(df: pd.DataFrame, col1: str, col2: str) -> t.Dict[t.Any, t.Any]:
    return df.set_index(col1)[col2].to_dict()


def column_or_index(df: pd.DataFrame, col: str) -> pd.Series | pd.Index:
    if col == "index":
        return df.index
    return df[col]


def intersect_columns(
    *dfs,
    cols: str | t.List[str],
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """"""
    if isinstance(cols, str):
        cols = [cols] * len(dfs)

    uniq_vals_columns = [set(column_or_index(d, c)) for d, c in zip(dfs, cols)]
    common_vals = set.intersection(*uniq_vals_columns)

    filtered_dfs = []
    for df, col in zip(dfs, cols):
        df = df[column_or_index(df, col).isin(common_vals)]
        filtered_dfs.append(df)

    return tuple(filtered_dfs)


def filter_by_value_counts(df: pd.DataFrame, col: t.Any, n: int) -> pd.DataFrame:
    """Filter a pd.DataFrame by value_counts in a given column."""
    counts = df[col].value_counts()
    keep_values = counts[counts >= n].index
    return df[df[col].isin(keep_values)]


def as_obs_df(
    df: pd.DataFrame,
    id_col: str = "id",
    label_col: str = "label",
    cell_id_col: str = "cell_id",
    drug_id_col: str = "drug_id",
) -> pd.DataFrame:
    """"""
    if id_col not in df.columns:
        df[id_col] = range(len(df))

    mapper = {
        id_col: "id",
        cell_id_col: "cell_id",
        drug_id_col: "drug_id",
        label_col: "label",
    }

    return df[list(mapper)].rename(columns=mapper)
