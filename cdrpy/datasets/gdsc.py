"""GDSC datasets"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

from collections import defaultdict
from types import SimpleNamespace

from cdrpy.constants import CDRPY_DATASET_BASE_URL
from cdrpy.datasets.base import CustomDataset, EncoderDict
from cdrpy.datasets.utils import ensure_dataset_download, extract_dataset_archive
from cdrpy.feat.encoders import PandasEncoder


_DESCRIPTION = "Genomics of Drug Sensitivity in Cancer (GDSC) dataset."


_SOURCES = SimpleNamespace(
    cell_gexp="cell_feature.gene_expression.log2tpm.csv",
    cell_meta="cell_metadata.csv",
    drug_meta="drug_metadata.csv",
    drug_smiles="drug_feature.smiles.csv",
    labels="labels.csv",
)


class GDSCDataset(CustomDataset):
    """GDSC dataset."""

    name = "gdsc"
    desc = _DESCRIPTION
    url = f"{CDRPY_DATASET_BASE_URL}/gdsc.tar.gz"

    def __init__(self, metric: t.Literal["auc", "ln_ic50"], *args, **kwargs) -> None:
        self.metric = metric
        super().__init__(*args, **kwargs)

    def download(self) -> None:
        """"""
        tarfile_path = ensure_dataset_download(url=self.url, data_dir=self.path)
        extract_dataset_archive(tarfile_path, extract_to=self.path)

    def read(
        self,
    ) -> t.Tuple[pd.DataFrame, EncoderDict, EncoderDict, pd.DataFrame, pd.DataFrame]:
        """"""
        obs = (
            pd.read_csv(self.joinpath(_SOURCES.labels))
            .filter(items=["id", "cell_id", "drug_id", self.metric])
            .rename(columns={self.metric: "label"})
        )

        cell_meta = pd.read_csv(self.joinpath(_SOURCES.cell_meta), index_col=0)
        drug_meta = pd.read_csv(self.joinpath(_SOURCES.drug_meta), index_col=0)

        cell_gexp_encoder = PandasEncoder.from_csv(
            self.joinpath(_SOURCES.cell_gexp),
            name="gexp",
            dtype=defaultdict(lambda: np.float32, cell_id=str),
            keep_default_na=False,
        )

        drug_smiles_encoder = PandasEncoder.from_csv(
            self.joinpath(_SOURCES.drug_smiles), name="smiles"
        )

        cell_encoders = {"exp": cell_gexp_encoder}
        drug_encoders = {"mol": drug_smiles_encoder}

        return obs, cell_encoders, drug_encoders, cell_meta, drug_meta
