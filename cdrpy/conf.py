"""

"""

from __future__ import annotations

import typing as t

from dataclasses import dataclass


@dataclass
class DatasetSourceConfig:
    """"""

    name: str
    label: str
    cell: dict[str, str]
    drug: dict[str, str]


@dataclass
class DatasetConfig:
    """"""

    name: str
    dir: str
    sources: DatasetSourceConfig
    label_args: dict[str, t.Any]
    split_subdir: str
    scale_labels_per_drug: bool


@dataclass
class ModelConfig:
    """"""

    name: str
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4


@dataclass
class OutputConfig:
    """"""

    out_dir: str
    log_dir: str
    save_model: bool = True


@dataclass
class Config:
    """"""

    dataset: DatasetConfig
    model: ModelConfig
    output: OutputConfig
