"""Data storage and download helpers.

FIXME: some of this should be moved to the cdrpy-data repo
NOTE: critical parts for cdrpy have been moved to cdrpy.datasets.utils
"""

from __future__ import annotations

import os
import logging

from urllib.parse import urlparse
from urllib.request import urlretrieve

from cdrpy.util.pbar import TQDMReportHook

from ...constants import (
    IS_WINDOWS_PLATFORM,
    CDRPY_DATA_PREFIX,
    DEFAULT_CDRPY_DATA_DIR,
    DEFAULT_CDRPY_DATASET_DIR,
)

log = logging.getLogger(__name__)


def home_dir() -> str:
    """Get user's home directory.

    FIXME: duplicate code with `utils/config.py`

    Uses %USERPROFILE% on windows, $HOME/getuid on POSIX.
    """
    if IS_WINDOWS_PLATFORM:
        return os.environ.get("USERPROFILE", "")
    else:
        return os.path.expanduser("~")


def cdrpy_data_dir_from_environment() -> str | None:
    """Get config path from environment variable."""
    data_dir = os.environ.get(CDRPY_DATA_PREFIX)
    if not data_dir:
        return None
    return os.path.join(data_dir, DEFAULT_CDRPY_DATA_DIR)


def cdrpy_dataset_dir_from_environment() -> str | None:
    """Get config path from environment variable."""
    dataset_dir = os.environ.get(CDRPY_DATA_PREFIX)
    if not dataset_dir:
        return None
    return os.path.join(dataset_dir, DEFAULT_CDRPY_DATASET_DIR)


def file_name_from_url(url: str) -> str:
    """"""
    # FIXME: need to properly parse all urls
    return os.path.basename(urlparse(url).path)


def find_cdrpy_data_dir(data_dir: str | None = None) -> str | None:
    """"""
    paths = list(
        filter(
            None,
            [
                data_dir,
                cdrpy_data_dir_from_environment(),
                os.path.join(home_dir(), DEFAULT_CDRPY_DATA_DIR),
            ],
        )
    )

    log.debug(f"Looking for cdrpy data directory at: {repr(paths)}")

    for path in paths:
        if os.path.exists(path):
            log.debug(f"Found cdrpy data directory at: {path}")
            return path

    log.debug("No cdrpy data directory found")

    raise ValueError("Could not find cdrpy data directory.")


def find_cdrpy_dataset_dir(dataset_dir: str | None = None) -> str | None:
    """"""
    paths = list(
        filter(
            None,
            [
                dataset_dir,
                cdrpy_dataset_dir_from_environment(),
                os.path.join(home_dir(), DEFAULT_CDRPY_DATASET_DIR),
            ],
        )
    )

    log.debug(f"Looking for cdrpy dataset directory at: {repr(paths)}")

    for path in paths:
        if os.path.exists(path):
            log.debug(f"Found cdrpy dataset directory at: {path}")
            return path

    log.debug("No cdrpy dataset directory found")

    raise ValueError("Could not find cdrpy dataset directory.")


def get_download_file_path(
    *subkeys, file_name: str, data_dir: str | None = None
) -> str:
    """"""
    data_dir = find_cdrpy_data_dir(data_dir)

    if data_dir is None:
        raise ValueError("Could not find cdrpy data directory.")

    return os.path.join(data_dir, *subkeys, file_name)


def safe_create_download_dir(dl_path: str) -> None:
    """Create download directory if it does not exist."""
    dl_root = os.path.dirname(dl_path)
    if not os.path.exists(dl_root):
        os.makedirs(dl_root)


def ensure_file_download(
    *subkeys: str,
    url: str,
    file_name: str | None = None,
    data_dir: str | None = None,
    force: bool = False,
) -> str:
    """"""
    data_dir = find_cdrpy_data_dir(data_dir)

    if data_dir is None:
        raise ValueError("Could not find cdrpy data directory.")

    if file_name is None:
        file_name = file_name_from_url(url)

    path = get_download_file_path(*subkeys, file_name=file_name, data_dir=data_dir)

    if os.path.isfile(path) and not force:
        # TODO: add hash check
        return path

    safe_create_download_dir(path)

    tqdm_kwargs = dict(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        disable=False,
        desc=f"Downloading {file_name}",
        leave=False,
    )

    with TQDMReportHook(**tqdm_kwargs) as t:
        urlretrieve(url, path, reporthook=t.update_to)

    return path


class DataStore:
    """"""

    def __init__(self, base: str | None = None, ensure_exists: bool = True) -> None:
        self.base = find_cdrpy_data_dir(base)
        if ensure_exists and not os.path.exists(self.base):
            os.makedirs(self.base)

    def ensure(
        self, *subkeys, url: str, file_name: str | None = None, force: bool = False
    ) -> str:
        """Ensures that a file has been downloaded."""
        if file_name is None:
            file_name = file_name_from_url(url)

        path = get_download_file_path(*subkeys, file_name=file_name, data_dir=self.base)

        safe_create_download_dir(path)

        if os.path.isfile(path) and not force:
            return path

        tqdm_kwargs = dict(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            disable=False,
            desc=f"Downloading {file_name}",
            leave=False,
        )

        with TQDMReportHook(**tqdm_kwargs) as t:
            path, _ = urlretrieve(url, path, reporthook=t.update_to)

        return path
