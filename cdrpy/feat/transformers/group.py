"""Sklearn-style transformers for transformations along groups."""

from __future__ import annotations

import pandas as pd
import numpy as np
import typing as t

from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)
from sklearn.preprocessing._data import _is_constant_feature, _handle_zeros_in_scale


def _safe_mean_and_var(X: np.ndarray) -> t.Tuple[np.ndarray]:
    """Safely calculate mean and variance."""
    if np.any(np.isnan(X)):
        mean_op = np.nanmean
        var_op = np.nanvar
    else:
        mean_op = np.mean
        var_op = np.var
    mean_ = mean_op(X, axis=0, dtype=np.float64)
    var_ = var_op(X, axis=0, dtype=np.float64)
    return mean_, var_


class GroupTransformerMixin(TransformerMixin):
    """Base class for all grouped transformers."""

    def fit_transform(
        self, X: pd.DataFrame | np.ndarray, y=None, groups=None, **fit_params
    ) -> np.ndarray:
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, groups=groups, **fit_params).transform(X, groups=groups)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, groups=groups, **fit_params).transform(
                X, groups=groups
            )


class GroupStandardScaler(BaseEstimator, GroupTransformerMixin, OneToOneFeatureMixin):
    """"""

    def __init__(self, *, copy: bool = True) -> None:
        self.copy = copy

    def _reset(self) -> None:
        """Reset initial data-dependent state of the scaler."""
        if hasattr(self, "scales_"):
            del self.groups_seen_
            del self.scales_
            del self.means_
            del self.vars_

    def fit(
        self, X: pd.DataFrame | np.ndarray, y=None, groups=None
    ) -> GroupStandardScaler:
        """Compute mean and std by group for later scaling."""
        self._reset()

        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        check_consistent_length(X, y, groups)
        X = self._validate_data(X, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        groups = column_or_1d(groups)
        unique_groups = np.unique(groups)
        self.groups_seen_ = set(unique_groups)

        self.scales_ = {}
        self.means_ = {}
        self.vars_ = {}
        for i in unique_groups:
            x_group = X[groups == i]
            group_mean, group_var = _safe_mean_and_var(x_group)

            self.means_[i] = group_mean
            self.vars_[i] = group_var

            constant_mask = _is_constant_feature(
                group_var, group_mean, x_group.shape[0]
            )
            self.scales_[i] = _handle_zeros_in_scale(
                np.sqrt(group_var), copy=False, constant_mask=constant_mask
            )

        return self

    def transform(
        self, X: pd.DataFrame | np.ndarray, groups: np.ndarray, copy=None
    ) -> np.ndarray:
        """Performs standardization by centering and scaling within groups."""
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = self._validate_data(
            X,
            reset=False,
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        groups = column_or_1d(groups)
        if any([g not in self.groups_seen_ for g in groups]):
            raise ValueError("Unseen group.")

        unique_groups = np.unique(groups)
        for i in unique_groups:
            X[groups == i] -= self.means_[i]
            X[groups == i] /= self.scales_[i]

        return X
