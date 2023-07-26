"""

"""

from __future__ import annotations

import typing as t

from dataclasses import dataclass
from pathlib import Path

from ..util import io


@dataclass
class Split:
    """"""

    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def load_split(split_dir: Path | str, id: int) -> Split:
    """Loads a single split from the specified folder."""
    split_dir = Path(split_dir)
    train_ids = io.read_pickled_list(split_dir / f"train_{id}.pickle")
    val_ids = io.read_pickled_list(split_dir / f"val_{id}.pickle")
    test_ids = io.read_pickled_list(split_dir / f"test_{id}.pickle")
    return Split(train_ids, val_ids, test_ids)


def load_splits(split_dir: Path | str) -> t.Generator[Split, None, None]:
    """"""
    split_dir = Path(split_dir)
    for i in range(10):
        train_ids = io.read_pickled_list(split_dir / f"train_{i+1}.pickle")
        val_ids = io.read_pickled_list(split_dir / f"val_{i+1}.pickle")
        test_ids = io.read_pickled_list(split_dir / f"test_{i+1}.pickle")
        yield Split(train_ids, val_ids, test_ids)
