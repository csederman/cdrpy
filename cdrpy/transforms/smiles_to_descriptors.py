"""Chemical structure-based descriptor transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

from rdkit import Chem
from rdkit.Chem import Descriptors

from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import DictEncoder
from cdrpy.transforms.base import DrugFeatureTransform


class SmilesToDescriptors(DrugFeatureTransform):
    """Converts SMILES strings to molecular descriptors."""

    _supported_encoders = DictEncoder  # NOTE: must be single type or tuple of types

    def __init__(self, descriptor_list: list[str] = None, **kwargs) -> None:
        """
        Initialize the transformer.

        Args:
            descriptor_list: List of descriptor names to compute. If None, uses all available descriptors.
        """
        self.descriptor_list = descriptor_list or self._get_all_descriptors()
        super().__init__(**kwargs)

    def _get_all_descriptors(self) -> list[str]:
        """Get list of all available RDKit descriptors."""
        return [desc[0] for desc in Descriptors._descList]

    def transform(self, D: Dataset) -> None:
        """Transform SMILES to molecular descriptors."""
        smiles_encoder = self.get_encoder(D)
        self.check_supported_encoder(smiles_encoder)

        descriptors = {}
        for key, smiles_str in smiles_encoder.data.items():
            desc_values = self.get_molecular_descriptors(smiles_str)
            descriptors[key] = desc_values

        # Convert to DataFrame for easier NaN checking
        desc_df = pd.DataFrame.from_dict(descriptors, orient="index")

        # Validate no NaN values
        self._validate_no_nan(desc_df)

        # Convert back to dictionary format for encoder
        descriptor_dict = desc_df.to_dict("index")
        # Convert each row to numpy array
        for key in descriptor_dict:
            descriptor_dict[key] = np.array(list(descriptor_dict[key].values()))

        descriptor_encoder = DictEncoder(descriptor_dict)
        self.set_encoder(D, descriptor_encoder)

    def get_molecular_descriptors(self, smiles_str: str) -> Dict[str, float]:
        """
        Calculate molecular descriptors for a SMILES string.

        Args:
            smiles_str: SMILES string

        Returns:
            Dictionary of descriptor names and values
        """
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles_str}")

        descriptors = {}
        for desc_name in self.descriptor_list:
            try:
                desc_func = getattr(Descriptors, desc_name)
                descriptors[desc_name] = desc_func(mol)
            except AttributeError:
                raise ValueError(f"Unknown descriptor: {desc_name}")

        return descriptors

    def _validate_no_nan(self, desc_df: pd.DataFrame) -> None:
        """
        Validate that there are no NaN values in the descriptor matrix.

        Args:
            desc_df: DataFrame containing descriptors

        Raises:
            ValueError: If NaN values are found
        """
        nan_mask = desc_df.isna()
        if nan_mask.any().any():
            nan_info = []
            for col in desc_df.columns:
                if nan_mask[col].any():
                    nan_count = nan_mask[col].sum()
                    nan_indices = desc_df.index[nan_mask[col]].tolist()
                    nan_info.append(
                        f"  {col}: {nan_count} NaN values at indices {nan_indices}"
                    )

            error_msg = f"NaN values found in descriptors:\n" + "\n".join(nan_info)
            raise ValueError(error_msg)
