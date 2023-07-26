"""

"""

from __future__ import annotations

import typing as t
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def check_pandas(X: t.Any) -> pd.DataFrame:
    """Checks that a variable is a `pd.DataFrame` instance."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a `pd.DataFrame` (got {type(X)})")
    return X


class PandasGroupedStandardScaler(BaseEstimator, TransformerMixin):
    """"""

    def __init__(self, value_col: str, group_col: str) -> None:
        self.value_col = value_col
        self.group_col = group_col

    def fit(self, X: t.Any, y: t.Any = None) -> PandasGroupedStandardScaler:
        """"""
        X = check_pandas(X)
        X_grouped = X.groupby(self.group_col)[self.value_col]
        self.groups = set(X_grouped.groups)

        means = X_grouped.mean()
        self.means = dict(zip(means.index, means.values))

        stds = X_grouped.std()
        self.stds = dict(zip(stds.index, stds.values))

        return self

    def transform(self, X: t.Any, y: t.Any = None) -> pd.DataFrame:
        """"""
        X = check_pandas(X)
        X_grouped = X.groupby(self.group_col)[self.value_col]

        self.check_groups(X_grouped.groups)

        X[self.value_col] = X_grouped.transform(
            lambda g: (g - self.means[g.name]) / self.stds[g.name]
        )

        return X

    def check_groups(self, groups: t.Iterable[t.Any]) -> None:
        """"""
        groups = set(groups)
        missing = groups.difference(self.groups)
        if len(missing) > 0:
            raise ValueError(f"Unknown groups ({missing})")


def make_grouped_standard_scaler_pandas(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
) -> t.Callable[[pd.DataFrame], pd.DataFrame]:
    """"""
    grouped = df.groupby(group_col)
    groups = list(grouped.groups)

    group_means = grouped[value_col].mean()
    group_means = dict(zip(group_means.index, group_means.values))

    group_stds = grouped[value_col].std()
    group_stds = dict(zip(group_stds.index, group_stds.values))

    def grouped_standard_scaler(df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(group_col)
        if not all(g in groups for g in grouped.groups):
            raise ValueError("can't transform values for unknown groups")
        df[value_col] = grouped[value_col].transform(
            lambda g: (g - group_means[g.name]) / group_stds[g.name]
        )

        return df

    return grouped_standard_scaler
