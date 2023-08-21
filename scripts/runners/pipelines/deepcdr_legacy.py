"""
Pipeline for running the original (legacy) DeepCDR code.

>>> DEEPCDR_ROOT="/scratch/ucgd/lustre-work/marth/u0871891/projects/cdrpy/pkg/DualGCN/code" \
        python scripts/runners/run.py \
            model=DualGCN-legacy \
            model.hyper.epochs=2 \
            dataset.preprocess.norm=global
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import sys
import tempfile

import numpy as np
import pandas as pd
import typing as t
import tensorflow as tf

from omegaconf import DictConfig
from pathlib import Path
from tensorflow import keras
from types import SimpleNamespace

from cdrpy.data.datasets import Dataset
from cdrpy.data.preprocess import normalize_responses
from cdrpy.feat.encoders import DictEncoder, RepeatEncoder
from cdrpy.metrics import tf_metrics
from cdrpy.splits import load_split
from cdrpy.util.io import read_pickled_dict
from cdrpy.util.validation import check_same_columns, check_same_indexes
