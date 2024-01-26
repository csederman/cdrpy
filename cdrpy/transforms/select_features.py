""""""

from __future__ import annotations

import typing as t

from cdrpy.transforms.base import FeatureTransform
from cdrpy.data.genelist import GenelistEnum, load_genelist
from cdrpy.feat.encoders import PandasEncoder

if t.TYPE_CHECKING:
    from cdrpy.datasets import Dataset


class SelectFeatures(FeatureTransform):
    """"""

    _supported_encoders = PandasEncoder

    def __init__(self, feature_names: t.Iterable[t.Any], **kwargs) -> None:
        self.feature_names = feature_names
        super().__init__(**kwargs)

    def transform(self, D: Dataset) -> None:
        # TODO: Add option to ensure all features are present
        encoder: PandasEncoder = self.get_encoder(D)
        self.check_supported_encoder(encoder)

        selected_data = encoder.data.filter(items=self.feature_names)
        selected_encoder = PandasEncoder(selected_data)

        self.set_encoder(D, selected_encoder)


class SelectCellFeatures(SelectFeatures):
    _feature_type = "cell"


class SelectDrugFeatures(SelectFeatures):
    _feature_type = "drug"


class SelectGenes(SelectCellFeatures):
    """"""

    def __init__(self, gene_list: GenelistEnum, **kwargs) -> None:
        feature_names = load_genelist(gene_list)
        super().__init__(feature_names=feature_names, **kwargs)
