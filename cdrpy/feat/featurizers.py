""""""

from __future__ import annotations

import pandas as pd
import typing as t

from rdkit import Chem
from rdkit.Chem import AllChem as AllChem


def morgan_fingerprint_from_smiles(smiles_strs: t.Iterable[str]) -> t.List[t.Any]:
    """"""
    mols = [Chem.MolFromSmiles(s) for s in smiles_strs]
    mfps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=512) for m in mols]

    return mfps


def make_morgan_fingerprint_features(
    df: pd.DataFrame, smiles_col: str, id_col: str | None = None
) -> pd.DataFrame:
    """"""
    if id_col is None:
        id_col = smiles_col

    drug_ids = df[id_col]
    smiles_strs = df[smiles_col]
    morgan_fps = morgan_fingerprint_from_smiles(smiles_strs)
    mol_feat_dict = dict(zip(drug_ids, list(map(list, morgan_fps))))

    return pd.DataFrame.from_dict(mol_feat_dict, orient="index")
