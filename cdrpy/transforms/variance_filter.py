""""""

from __future__ import annotations

import typing as t

from cdrpy.transforms.base import FeatureTransform
from cdrpy.feat.encoders import PandasEncoder

if t.TYPE_CHECKING:
    from cdrpy.datasets import Dataset


class VarianceFilter(FeatureTransform):
    """"""

    _supported_encoders = PandasEncoder

    def __init__(self, variance_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.variance_threshold = variance_threshold

    def transform(self, D: Dataset) -> None:
        encoder: PandasEncoder = self.get_encoder(D)
        self.check_supported_encoder(encoder)

        variances = encoder.data.var()
        keep_features = variances[variances > self.variance_threshold].index

        self.set_encoder(D, PandasEncoder(encoder.data[keep_features]))


class CellFeatureVarianceFilter(VarianceFilter):
    _feature_type = "cell"


class DrugFeatureVarianceFilter(VarianceFilter):
    _feature_type = "drug"
