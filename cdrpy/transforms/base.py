""""""

from __future__ import annotations

import typing as t

from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from cdrpy.datasets import Dataset, EncoderDict
    from cdrpy.feat.encoders import Encoder


class Transform(ABC):
    """Base class from which all dataset transformations will inherit."""

    def __call__(self, D: Dataset) -> None:
        self.transform(D)

    @abstractmethod
    def transform(self, D: Dataset) -> None:
        pass


class FeatureTransform(Transform):
    """"""

    _feature_type: t.Literal["cell", "drug"]
    _supported_encoders: t.Union[Encoder, t.Tuple[Encoder]]

    def __init__(self, key: t.Any) -> None:
        self.key = key

    def _encoders(self, D) -> EncoderDict:
        if self._feature_type == "cell":
            return D.cell_encoders
        elif self._feature_type == "drug":
            return D.drug_encoders
        else:
            raise ValueError(f"Expected 'cell' or 'drug', got {self._feature_type}")

    def get_encoder(self, D: Dataset) -> Encoder:
        return self._encoders(D)[self.key]

    def set_encoder(self, D: Dataset, E: Encoder) -> None:
        self._encoders(D)[self.key] = E

    def _is_supported_encoder(self, E: Encoder) -> bool:
        """Check whether an encoder class is supported."""
        if self._supported_encoders is None:
            raise ValueError("Supported encoders must be set.")
        return isinstance(E, self._supported_encoders)

    def check_supported_encoder(self, E: Encoder) -> None:
        if not self._is_supported_encoder(E):
            raise TypeError(
                f"Expected an instance of {self._supported_encoders}, got {type(E)}"
            )


class CellFeatureTransform(FeatureTransform):
    _feature_type = "cell"


class DrugFeatureTransform(FeatureTransform):
    _feature_type = "drug"
