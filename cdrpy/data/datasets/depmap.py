""""""

from __future__ import annotations

import os
import functools
import requests

import pandas as pd
import typing as t

from cdrpy.feat.encoders import PandasEncoder
from cdrpy.data.utils.store import DataStore
from cdrpy.types import JSONObject
from cdrpy.feat.featurizers import make_morgan_fingerprint_features

from .base import CustomDataset, EncoderDict
from ..utils import pandas as pd_utils


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


def get_depmap_download_url(dl_table: JSONObject, file_name: str, release: str) -> str:
    """Parse release table and extract matching download URL."""
    available_files = dl_table["table"]
    for file in available_files:
        if file["fileName"] == file_name and file["releaseName"] == release:
            return file["downloadUrl"]
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


# FIXME: move this function to proper location
def filter_by_column_values(
    *dfs, cols: str | t.List[str]
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """"""
    if isinstance(cols, str):
        cols = [cols] * len(dfs)

    uniq_vals_columns = [set(df[col]) for df, col in zip(dfs, cols)]
    common_vals = set.intersection(*uniq_vals_columns)

    filtered_dfs = []
    for df, col in zip(dfs, cols):
        df = df[df[col].isin(common_vals)]
        filtered_dfs.append(df)

    return tuple(filtered_dfs)


class DepMapDataset(CustomDataset):
    source = "DepMap"


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

    def __init__(self, gene_list: t.List[str] | None = None, **kwargs) -> None:
        self.gene_list = gene_list
        super().__init__(**kwargs)

    def download(self) -> None:
        """Downloads the raw PRISM response data."""
        os.makedirs(self.path, exist_ok=True)

        # download the raw data
        store = DataStore()  # NOTE: singleton? get_or_create_store?
        dl_table = get_depmap_downloads_table()

        file_dict = {}
        for id_, release, name in DEPMAP_PRISM_REPURPOSING_MANIFEST:
            url = get_depmap_download_url(dl_table, name, release)
            file_dict[id_] = store.ensure(self.source, release, url=url, file_name=name)

        self.preprocess(file_dict)

    def preprocess(self, file_dict: t.Dict[str, str]) -> None:
        """Optional function called at the end of download."""
        smiles_data = read_raw_prism_smiles_data(
            file_dict["drug_smiles_primary"], file_dict["drug_smiles_secondary"]
        )

        id_to_smiles = smiles_data[["broad_id", "smiles"]].drop_duplicates().dropna()
        id_to_smiles["smiles"] = id_to_smiles["smiles"].map(lambda s: s.split(",")[0])
        id_to_smiles = id_to_smiles.set_index("broad_id")["smiles"].to_dict()

        cell_meta = pd.read_csv(file_dict["cell_meta"])
        drug_meta = read_raw_prism_drug_meta(file_dict["drug_meta"], id_to_smiles)
        resp_data = read_raw_prism_response_data(file_dict["resp_data"])
        gexp_data = read_raw_depmap_expression_data(
            file_dict["gexp_data"], self.gene_list
        )

        gexp_data, cell_meta, resp_data = pd_utils.intersect_columns(
            gexp_data, cell_meta, resp_data, cols=["index", "ModelID", "depmap_id"]
        )
        cell_meta = pd_utils.column_as_index(cell_meta, "ModelID", "cell_id")

        resp_data, drug_meta = pd_utils.intersect_columns(
            resp_data, drug_meta, cols=["drug_id", "IDs"]
        )
        drug_id_mapping = pd_utils.dict_from_columns(drug_meta, "IDs", "broad_id")
        resp_data["drug_id"] = resp_data["drug_id"].map(drug_id_mapping)

        # generate morgan fingerprint features
        chem_data = make_morgan_fingerprint_features(drug_meta, "smiles", "broad_id")
        drug_meta = pd_utils.column_as_index(drug_meta, "broad_id", "drug_id")

        # format response data
        resp_data = pd_utils.as_obs_df(
            resp_data, label_col="LFC", cell_id_col="depmap_id"
        )

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
        exp_enc = PandasEncoder.from_csv(exp_feat_path)
        mol_enc = PandasEncoder.from_csv(mol_feat_path)
        cell_encoders = {"exp": exp_enc}
        drug_encoders = {"mol": mol_enc}

        obs_path = os.path.join(self.path, "labels.csv")
        obs = pd.read_csv(obs_path)

        cell_meta_path = os.path.join(self.path, "cell-annotations.csv")
        drug_meta_path = os.path.join(self.path, "drug-annotations.csv")
        cell_meta = pd.read_csv(cell_meta_path, index_col=0)
        drug_meta = pd.read_csv(drug_meta_path, index_col=0)

        return obs, cell_encoders, drug_encoders, cell_meta, drug_meta
