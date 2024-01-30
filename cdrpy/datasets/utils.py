"""Dataset utilities."""

from __future__ import annotations

import os
import logging
import tarfile

from urllib.parse import urlparse
from urllib.request import urlretrieve

from cdrpy.constants import (
    DEFAULT_CDRPY_DATASET_DIR,
    IS_WINDOWS_PLATFORM,
    CDRPY_DATA_PREFIX,
)
from cdrpy.util.pbar import TQDMReportHook, get_tqdm_default_params


log = logging.getLogger(__name__)


def home_dir() -> str:
    """Get user's home directory.

    Uses %USERPROFILE% on windows, $HOME/getuid on POSIX.
    """
    if IS_WINDOWS_PLATFORM:
        return os.environ.get("USERPROFILE", "")
    else:
        return os.path.expanduser("~")


def cdrpy_dataset_dir_from_environment() -> str | None:
    """Get config path from environment variable."""
    dataset_dir_prefix = os.environ.get(CDRPY_DATA_PREFIX)

    if dataset_dir_prefix is None:
        return None

    return os.path.join(dataset_dir_prefix, DEFAULT_CDRPY_DATASET_DIR)


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


def file_name_from_url(url: str) -> str:
    """"""
    # FIXME: need to properly parse all urls
    return os.path.basename(urlparse(url).path)


def safe_create_download_dir(dl_path: str) -> None:
    """Create download directory if it does not exist."""
    root = os.path.dirname(dl_path)
    if not os.path.exists(root):
        os.makedirs(root)


def ensure_dataset_download(
    url: str,
    data_dir: str,
    file_name: str | None = None,
    force: bool = False,
) -> str:
    """"""
    if file_name is None:
        file_name = file_name_from_url(url)

    download_path = os.path.join(data_dir, file_name)

    if os.path.isfile(download_path) and not force:
        # FIXME: add hash check
        return download_path

    safe_create_download_dir(download_path)

    with TQDMReportHook(**get_tqdm_default_params(file_name)) as t:
        urlretrieve(url, download_path, reporthook=t.update_to)

    return download_path


def extract_dataset_archive(tarfile_path: str, extract_to: str) -> None:
    """"""
    with tarfile.open(tarfile_path, "r:gz") as tar:
        tar.extractall(extract_to)
