#!/usr/bin/env python

from __future__ import annotations

import click
import requests
import time

import polars as pl

from pathlib import Path
from tqdm import tqdm


properties = ",".join(
    [
        "CanonicalSMILES",
        "InChIKey",
        "MolecularFormula",
        "MolecularWeight",
        "Title",
    ]
)

url_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
url_fmt = f"{url_base}/{{}}/property/{properties}/JSON"


@click.command()
@click.option("--output-dir", help="Output directory.")
def main() -> None:
    gdsc_data_folder = Path("../../data/raw/GDSC")
    gdsc_drug_list = pl.read_csv(
        gdsc_data_folder / "drug_list_2023_06_23_curation.csv",
        dtypes={
            "Drug Id": int,
            "PubCHEM": str,
            "PubCHEM__curation": int,
            "InChiKey__curation": str,
            "confidence__curation": str,
            "notes__curation": str,
        },
    )

    pubchem_cids = (
        gdsc_drug_list["PubCHEM__curation"].drop_nulls().unique().to_list()
    )

    results = []
    n_req = 0
    for cid in tqdm(pubchem_cids, desc="Fetching PubCHEM properties"):
        resp = requests.get(url_fmt.format(cid))
        try:
            json_ = resp.json()
            results.append(json_["PropertyTable"]["Properties"][0])
        except Exception as e:
            print(e)
            pass
        n_req += 1
        if n_req % 5 == 0:
            time.sleep(0.5)  # avoid rate limiting errors

    results = pl.DataFrame(results)
    results.columns = [f"{c}__PubCHEM" for c in results.columns]

    gdsc_drug_list_annotated = gdsc_drug_list.join(
        results,
        left_on="PubCHEM__curation",
        right_on="CID__PubCHEM",
        how="left",
    )

    gdsc_drug_list_annotated.write_csv(
        gdsc_data_folder / "drug_list_2023_06_23_curation_annotated.csv"
    )


if __name__ == "__main__":
    main()
