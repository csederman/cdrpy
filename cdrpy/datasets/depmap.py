""""""

from __future__ import annotations

import pandas as pd

from cdrpy.constants import CDRPY_DATASET_BASE_URL

from cdrpy.datasets.base import CustomDataset
from cdrpy.datasets.utils import ensure_dataset_download, extract_dataset_archive
from cdrpy.feat.encoders import PandasEncoder


class DepMapDataset(CustomDataset):
    """Base class from which all DepMap datasets will inherit."""

    def download(self) -> None:
        """"""
        tarfile_path = ensure_dataset_download(url=self.url, data_dir=self.path)
        extract_dataset_archive(tarfile_path, extract_to=self.path)


PRISM_REPURPOSING_DESCRIPTION = ""


class PRISMRepurposingDataset(DepMapDataset):
    """"""

    name = "depmap-prism-repurposing"
    desc = PRISM_REPURPOSING_DESCRIPTION
    url = f"{CDRPY_DATASET_BASE_URL}/depmap-prism-repurposing.tar.gz"

    def read(self) -> None:
        """"""
        obs = pd.read_csv(self.joinpath("labels.csv"))

        cell_enc = {"exp": PandasEncoder.from_csv(self.joinpath("cell_feature_ge.csv"))}
        drug_enc = {"mol": PandasEncoder.from_csv(self.joinpath("drug_feature_fp.csv"))}

        cell_meta = pd.read_csv(self.joinpath("cell_metadata.csv"), index_col=0)
        drug_meta = pd.read_csv(self.joinpath("drug_metadata.csv"), index_col=0)

        return obs, cell_enc, drug_enc, cell_meta, drug_meta
