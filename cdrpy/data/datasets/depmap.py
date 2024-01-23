""""""

from __future__ import annotations

import os
import functools
import requests

import pandas as pd
import typing as t

from cdrpy.feat.encoders import PandasEncoder
from cdrpy.data.utils.store import ensure_file_download
from cdrpy.types import JSONObject
from cdrpy.feat.featurizers import make_morgan_fingerprint_features

from .base import CustomDataset, EncoderDict
from ..utils import pandas as pdu
from .._base import Genelist, load_genelist


DEPMAP_API_BASE = "https://depmap.org/portal/download/api/downloads"


@functools.lru_cache(maxsize=1)
def get_depmap_downloads_table() -> JSONObject:
    """Fetch the  DepMap downloads table."""
    resp = requests.get(DEPMAP_API_BASE)
    resp.raise_for_status()
    return resp.json()


@functools.lru_cache(maxsize=1)
def get_latest_depmap_release(dl_table: JSONObject) -> str:
    """Parse latest release from DepMap."""
    release_data = dl_table["releaseData"]
    for release in release_data:
        if release["isLatest"] is True:
            return release["releaseName"]
    raise ValueError("No latest release found.")


def get_depmap_releases(dl_table: JSONObject) -> t.List[str]:
    """Returns a list of DepMap releases."""
    return [r["releaseName"] for r in dl_table["releaseData"]]


def fix_depmap_url(url: str | None) -> str | None:
    """"""
    if isinstance(url, str):
        if url.startswith("/"):
            url = "https://depmap.org" + url
    return url


def get_depmap_download_url(
    dl_table: JSONObject, file_name: str, release: str
) -> str | None:
    """Parse release table and extract matching download URL."""
    available_files = dl_table["table"]
    for file in available_files:
        if file["fileName"] == file_name and file["releaseName"] == release:
            url = fix_depmap_url(file["downloadUrl"])
            return url
    raise ValueError("File not found.")


def read_raw_prism_response_data(path: str) -> pd.DataFrame:
    """"""
    return (
        pd.read_csv(path, index_col=0)
        .melt(ignore_index=False, var_name="depmap_id", value_name="LFC")
        .rename_axis(index="drug_id")
        .reset_index()
        .dropna()
    )


def read_raw_prism_drug_meta(
    path: str, broad_id_to_smiles: t.Dict[str, str]
) -> pd.DataFrame:
    """"""
    df = pd.read_csv(path)

    df["broad_id"] = df["IDs"].map(lambda x: x.split(":")[-1])
    df["smiles"] = df["broad_id"].map(broad_id_to_smiles)
    df = df.dropna(subset="smiles")

    # drop duplicates and drugs outside of dose threshold (2.5 +/- 0.5)
    df["rel_dose"] = (df["dose"] - 2.5).abs()
    df = df[df["rel_dose"] < 0.5]
    df = df.sort_values("rel_dose").drop_duplicates("smiles", keep="first")

    # select a single smiles per drug name
    df = df.drop_duplicates("Drug.Name")

    return df


def read_raw_depmap_expression_data(
    path: str, gene_list: t.List[str] | None = None
) -> pd.DataFrame:
    """"""
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.map(lambda x: x.split(" ")[0])

    if gene_list is not None:
        keep_genes = df.columns.intersection(gene_list)
        df = df[keep_genes]

    return df


def read_raw_prism_smiles_data(*paths: str) -> pd.DataFrame:
    """"""
    return pd.concat(list(map(pd.read_csv, paths)))


class DepMapDataset(CustomDataset):
    """Base class from which all DepMap datasets will inherit."""

    source = "depmap"
    manifest = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _download_depmap_files(self) -> t.Dict[str, str]:
        """Downloads data files from DepMap."""
        dl_table = get_depmap_downloads_table()
        dl_paths = {}

        for id_, release, name in self.manifest:
            url = get_depmap_download_url(dl_table, name, release)
            if url is None:
                raise ValueError(f"No download url for {name}")
            dl_path = ensure_file_download(
                self.source, release, url=url, file_name=name
            )
            dl_paths[id_] = dl_path

        return dl_paths


DEPMAP_PRISM_REPURPOSING_MANIFEST = [
    (
        "drug_meta",
        "PRISM Repurposing Public 23Q2",
        "Repurposing_Public_23Q2_Extended_Primary_Compound_List.csv",
    ),
    (
        "resp_data",
        "PRISM Repurposing Public 23Q2",
        "Repurposing_Public_23Q2_Extended_Primary_Data_Matrix.csv",
    ),
    (
        "gexp_data",
        "DepMap Public 23Q4",
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    ),
    (
        "cell_meta",
        "DepMap Public 23Q4",
        "Model_v2.csv",
    ),
    (
        "drug_smiles_primary",
        "PRISM Repurposing 19Q4",
        "primary-screen-replicate-collapsed-treatment-info.csv",
    ),
    (
        "drug_smiles_secondary",
        "PRISM Repurposing 19Q4",
        "secondary-screen-replicate-collapsed-treatment-info.csv",
    ),
]


class PRISMRepurposingDataset(DepMapDataset):
    """"""

    name = "depmap-prism-repurposing"
    manifest = DEPMAP_PRISM_REPURPOSING_MANIFEST

    def download(self) -> None:
        """Downloads the raw PRISM response data."""
        os.makedirs(self.path, exist_ok=True)
        file_dict = self._download_depmap_files()
        self.preprocess(file_dict)

    def _load_raw_data(self, file_dict: t.Dict[str, str]) -> None:
        """"""
        smiles_data = read_raw_prism_smiles_data(
            file_dict["drug_smiles_primary"], file_dict["drug_smiles_secondary"]
        )

        id_to_smiles = smiles_data[["broad_id", "smiles"]].drop_duplicates().dropna()
        id_to_smiles["smiles"] = id_to_smiles["smiles"].map(lambda s: s.split(",")[0])
        id_to_smiles = pdu.dict_from_columns(id_to_smiles, "broad_id", "smiles")

        cell_meta = pd.read_csv(file_dict["cell_meta"])
        drug_meta = read_raw_prism_drug_meta(file_dict["drug_meta"], id_to_smiles)
        resp_data = read_raw_prism_response_data(file_dict["resp_data"])

        gene_list = load_genelist(Genelist.HALLMARK)
        gexp_data = read_raw_depmap_expression_data(file_dict["gexp_data"], gene_list)

        return cell_meta, drug_meta, resp_data, gexp_data

    def preprocess(self, file_dict: t.Dict[str, str]) -> None:
        """Optional function called at the end of download."""
        cell_meta, drug_meta, resp_data, gexp_data = self._load_raw_data(file_dict)

        gexp_data, cell_meta, resp_data = pdu.intersect_columns(
            gexp_data, cell_meta, resp_data, cols=["index", "ModelID", "depmap_id"]
        )
        cell_meta = pdu.column_as_index(cell_meta, "ModelID", "cell_id")

        resp_data, drug_meta = pdu.intersect_columns(
            resp_data, drug_meta, cols=["drug_id", "IDs"]
        )
        drug_id_mapping = pdu.dict_from_columns(drug_meta, "IDs", "broad_id")
        resp_data["drug_id"] = resp_data["drug_id"].map(drug_id_mapping)

        # generate morgan fingerprint features
        chem_data = make_morgan_fingerprint_features(drug_meta, "smiles", "broad_id")
        drug_meta = pdu.column_as_index(drug_meta, "broad_id", "drug_id")

        # remove cell lines wiht fewer than 50 drugs screened
        resp_data = pdu.filter_by_value_counts(resp_data, "depmap_id", 50)
        keep_cells = sorted(list(set(resp_data["depmap_id"])))

        cell_meta = cell_meta.loc[keep_cells]
        gexp_data = gexp_data.loc[keep_cells]

        # format response data
        resp_data = pdu.as_obs_df(resp_data, label_col="LFC", cell_id_col="depmap_id")

        # save the processed data files
        resp_data.to_csv(os.path.join(self.path, "labels.csv"), index=False)
        chem_data.to_csv(os.path.join(self.path, "morgan-fingerprint-features.csv"))
        gexp_data.to_csv(os.path.join(self.path, "gene-expression-features.csv"))
        drug_meta.to_csv(os.path.join(self.path, "drug-annotations.csv"))
        cell_meta.to_csv(os.path.join(self.path, "cell-annotations.csv"))

    def read(
        self,
    ) -> t.Tuple[pd.DataFrame, EncoderDict, EncoderDict, pd.DataFrame, pd.DataFrame]:
        """"""
        exp_feat_path = os.path.join(self.path, "gene-expression-features.csv")
        mol_feat_path = os.path.join(self.path, "morgan-fingerprint-features.csv")
        cell_encoders = {"exp": PandasEncoder.from_csv(exp_feat_path)}
        drug_encoders = {"mol": PandasEncoder.from_csv(mol_feat_path)}

        obs_path = os.path.join(self.path, "labels.csv")
        obs = pd.read_csv(obs_path)

        cell_meta_path = os.path.join(self.path, "cell-annotations.csv")
        drug_meta_path = os.path.join(self.path, "drug-annotations.csv")
        cell_meta = pd.read_csv(cell_meta_path, index_col=0)
        drug_meta = pd.read_csv(drug_meta_path, index_col=0)

        return obs, cell_encoders, drug_encoders, cell_meta, drug_meta
