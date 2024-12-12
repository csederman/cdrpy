import unittest

import pandas as pd

from cdrpy.datasets.base import Dataset

from . import fake_dataset


N_CELLS = 10
N_DRUGS = 10
N_RESPONSES = N_CELLS * N_DRUGS


class DatasetTest(unittest.TestCase):
    """"""

    def setUp(self) -> None:
        self.obs_data = fake_dataset.get_fake_data(N_CELLS, N_DRUGS)
        self.dataset = Dataset(self.obs_data)

    def test_init(self) -> None:
        self.assertEqual(self.dataset.obs.shape, (N_RESPONSES, 4))
        self.assertIsNone(self.dataset.cell_encoders)
        self.assertIsNone(self.dataset.drug_encoders)
        self.assertIsNone(self.dataset.cell_meta)
        self.assertIsNone(self.dataset.drug_meta)
        self.assertFalse(self.dataset.encode_drugs_first)
        self.assertIsNone(self.dataset.name)
        self.assertIsNone(self.dataset.desc)

    def test_validate_obs(self) -> None:
        with self.assertRaises(ValueError):
            self.dataset._validate_obs(pd.DataFrame({"id": [1, 2, 3]}))

    def test_dtype(self) -> None:
        self.assertEqual(self.dataset.dtype.name, "float64")

    def test_size(self) -> None:
        self.assertEqual(self.dataset.size, N_RESPONSES)

    def test_n_cells(self) -> None:
        self.assertEqual(self.dataset.n_cells, N_CELLS)

    def test_n_drugs(self) -> None:
        self.assertEqual(self.dataset.n_drugs, N_DRUGS)
