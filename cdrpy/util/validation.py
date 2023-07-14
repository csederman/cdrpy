"""

"""

from __future__ import annotations

import pandas as pd


def check_same_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    assert df1.columns.to_list() == df2.columns.to_list()


def check_same_indexes(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    assert df1.index.to_list() == df2.index.to_list()
