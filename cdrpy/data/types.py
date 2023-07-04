"""
Typing utilites for datasets.
"""

from __future__ import annotations

import numpy as np
import typing as t


EncodedFeatures = t.List[np.ndarray]
EncodedDataset = t.Tuple[EncodedFeatures, np.ndarray, np.ndarray, np.ndarray]
