"""Global variables."""

import sys

# environment variables
CDRPY_DATA_PREFIX = "CDRPY_DATA_PREFIX"

# data storage
DEFAULT_CDRPY_DATA_DIR = ".cdrpy/data"
DEFAULT_CDRPY_DATASET_DIR = ".cdrpy/datasets"

# system
IS_WINDOWS_PLATFORM = sys.platform == "win32"