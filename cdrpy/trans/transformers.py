"""

"""

from __future__ import annotations

import pickle
import random

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

keras = tf.keras  # pylance issue #1066

from keras import Model
from keras import layers, losses, optimizers
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_array

from dataclasses import dataclass
from pathlib import Path


class GroupedStandardScaler(BaseEstimator, TransformerMixin):
    """"""

    def __init__(
        self,
        *,
        copy: bool = True,
        use_median: bool = False,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        self.use_median = use_median
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X, y=None, groups=None) -> GroupedStandardScaler:
        """"""
        self.group_center: dict[str, float] = {}
        self.group_std: dict[str, float] = {}

        groups = check_array(
            groups, input_name="groups", ensure_2d=False, dtype=None
        )

        sorted_groups_idx = np.argsort(groups)
        sorted_groups = groups[sorted_groups_idx]

        uniq_groups, group_idx = np.unique(sorted_groups, return_index=True)
        self.groups = uniq_groups.tolist()

        X_grouped = np.split(X[sorted_groups_idx], group_idx[1:])
        center_func = np.median if self.use_median else np.mean
        for group_id, group in zip(uniq_groups, X_grouped):
            self.group_center[group_id] = center_func(group)
            self.group_std[group_id] = np.std(group)

        return self

    def transform(self, X, y=None, groups=None) -> np.ndarray:
        """"""
        groups = check_array(
            groups,
            input_name="groups",
            ensure_2d=False,
            dtype=None,
        )
        center_arr = np.array([self.group_center[g] for g in groups])
        std_arr = np.array([self.group_std[g] for g in groups])

        center_arr = center_arr.reshape(-1, 1)
        std_arr = std_arr.reshape(-1, 1)

        return (X - center_arr) / std_arr
