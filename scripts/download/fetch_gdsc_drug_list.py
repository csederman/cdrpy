#!/usr/bin/env python

import click
import requests

import polars as pl

from datetime import datetime
from io import StringIO
from pathlib import Path


url = "https://www.cancerrxgene.org/api/compounds"


@click.command()
@click.option("--output-dir", help="Output directory.")
def main(output_dir: str) -> None:
    """Fetch GDSCv2 compounds list."""
    ts = datetime.now().strftime("%Y_%m_%d")

    with requests.Session() as s:
        resp = s.get(url, params={"list": "all", "export": "csv"})
        content = resp.content.decode()
        GDSC_drug_list = pl.read_csv(StringIO(content))

    GDSC_drug_list.columns = [c.strip() for c in GDSC_drug_list.columns]
    GDSC_drug_list.write_csv(Path(output_dir) / f"drug_list_{ts}.csv")


if __name__ == "__main__":
    main()
