#!/usr/bin/env python
"""
Testing for hydra configurations.

NOTE: What I should do is have a bunch of bash scripts to run the various
    multirun sweep experiments with all the sweep params. These scripts
    can override the job dir so that we have control over the output dirs.
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import importlib
import logging
import random

import numpy as np
import tensorflow as tf

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from omegaconf import DictConfig


log = logging.getLogger(__name__)


PIPELINES = {
    "ScreenDL": "screendl",
    "DualGCN": "dualgcn",
}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """"""
    # What I should do here is just use importlib
    if not cfg.model.name in PIPELINES:
        raise ValueError("Unsupported model.")

    module_file = PIPELINES[cfg.model.name]
    module_path = f"pipelines.{module_file}"
    module = importlib.import_module(module_path)
    module.run_pipeline(cfg)


if __name__ == "__main__":
    run()
