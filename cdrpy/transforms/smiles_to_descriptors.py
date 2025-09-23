"""Chemical structure-based descriptor transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from typing import Dict

from rdkit import Chem
from rdkit.Chem import Descriptors

from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import DictEncoder
from cdrpy.transforms.base import DrugFeatureTransform


class SmilesToDescriptors(DrugFeatureTransform):
    """Converts SMILES strings to molecular descriptors."""

    _supported_encoders = DictEncoder  # NOTE: must be single type or tuple of types

    def __init__(
        self, descriptor_list: list[str] = None, min_desc: int = 10, **kwargs
    ) -> None:
        """
        Initialize the transformer.

        Args:
            descriptor_list: List of descriptor names to compute. If None, uses all available descriptors.
            min_desc: Minimum number of descriptors required after dropping NaN columns.
        """
        self.descriptor_list = descriptor_list or self._get_all_descriptors()
        self.min_desc = min_desc
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

        # Convert to DataFrame for easier NaN handling
        desc_df = pd.DataFrame.from_dict(descriptors, orient="index")

        # Drop columns with NaN values and warn
        desc_df_clean = self._drop_nan_columns(desc_df)

        # Check if we have enough features remaining
        if desc_df_clean.shape[1] < self.min_desc:
            raise ValueError(
                f"Too few descriptors remaining after dropping NaN columns. "
                f"Have {desc_df_clean.shape[1]}, need at least {self.min_desc}."
            )

        # Convert back to dictionary format for encoder
        descriptor_dict = desc_df_clean.to_dict("index")
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

    def _drop_nan_columns(self, desc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with NaN values and warn about dropped columns.

        Args:
            desc_df: DataFrame containing descriptors

        Returns:
            DataFrame with NaN columns removed
        """
        initial_cols = desc_df.shape[1]
        nan_cols = desc_df.columns[desc_df.isna().any()].tolist()

        if nan_cols:
            desc_df_clean = desc_df.drop(columns=nan_cols)
            dropped_count = len(nan_cols)
            remaining_count = desc_df_clean.shape[1]

            warnings.warn(
                f"Dropped {dropped_count} descriptor columns with NaN values: {nan_cols}. "
                f"Remaining descriptors: {remaining_count}/{initial_cols}",
                UserWarning,
            )
        else:
            desc_df_clean = desc_df

        return desc_df_clean
