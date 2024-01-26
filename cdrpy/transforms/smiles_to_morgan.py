"""Chemical structure-based feature transformers."""

from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem as AllChem

from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import DictEncoder
from cdrpy.transforms.base import DrugFeatureTransform


class SmilesToMorgan(DrugFeatureTransform):
    """Converts SMILES strings to Morgan fingerprints."""

    _supported_encoders = DictEncoder  # NOTE: must be single type or tuple of types

    def __init__(self, n_bits: int = 512, radius: int = 3, **kwargs) -> None:
        self.n_bits = n_bits
        self.radius = radius
        super().__init__(**kwargs)

    def transform(self, D: Dataset) -> None:
        """"""
        smiles_encoder = self.get_encoder(D)
        self.check_supported_encoder(smiles_encoder)

        fingerprints = map(self.get_morgan_fingerprint, smiles_encoder.data.values())
        fingerprint_dict = dict(zip(smiles_encoder.data.keys(), fingerprints))
        fingerprint_encoder = DictEncoder(fingerprint_dict)

        self.set_encoder(D, fingerprint_encoder)

    def get_morgan_fingerprint(self, smiles_str: str) -> np.ndarray[int]:
        """"""
        mol = Chem.MolFromSmiles(smiles_str)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        return np.array(list(fp))
