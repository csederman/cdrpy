import itertools

import pandas as pd
import numpy as np

from cdrpy.datasets.base import Dataset


def get_fake_data(n_cells: int, n_drugs: int) -> pd.DataFrame:
    """Simulates fake drug response data."""
    n_responses = n_cells * n_drugs

    cell_ids = [f"C{i+1}" for i in range(n_cells)]
    drug_ids = [f"D{i+1}" for i in range(n_drugs)]

    pairs = zip(*itertools.product(cell_ids, drug_ids))
    data = zip(range(n_responses), *pairs, np.random.normal(size=n_responses))

    return pd.DataFrame(data, columns=["id", "cell_id", "drug_id", "label"])
