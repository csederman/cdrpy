""""""

from __future__ import annotations

import pandas as pd
import typing as t

from cdrpy.constants import CDRPY_DATASET_BASE_URL

from cdrpy.datasets.base import CustomDataset
from cdrpy.datasets.utils import ensure_dataset_download, extract_dataset_archive
from cdrpy.feat.encoders import PandasEncoder


if t.TYPE_CHECKING:
    from .base import EncoderDict


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

    def read(self, featureless: bool = False) -> t.Tuple[
        pd.DataFrame,
        EncoderDict | None,
        EncoderDict | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
    ]:
        """"""
        obs = pd.read_csv(self.joinpath("labels.csv"))
        if featureless:
            return obs, None, None, None, None

        ge_enc = PandasEncoder.from_csv(
            self.joinpath("cell_feature_ge.csv"), name="ge_encoder"
        )
        fp_enc = PandasEncoder.from_csv(
            self.joinpath("drug_feature_fp.csv"), name="fp_encoder"
        )

        cell_encoders = {"exp": ge_enc}
        drug_encoders = {"mol": fp_enc}

        cell_meta = pd.read_csv(self.joinpath("cell_metadata.csv"), index_col=0)
        drug_meta = pd.read_csv(self.joinpath("drug_metadata.csv"), index_col=0)

        return obs, cell_encoders, drug_encoders, cell_meta, drug_meta
