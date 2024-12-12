import os
import unittest
import shutil
import tempfile
import json

from pytest import mark, fixture
from unittest import mock

from cdrpy.datasets.utils import find_cdrpy_dataset_dir
from cdrpy.constants import CDRPY_DATA_PREFIX, DEFAULT_CDRPY_DATASET_DIR

import logging

log = logging.getLogger(__name__)


class FindDatasetDirTest(unittest.TestCase):
    @fixture(autouse=True)
    def tmpdir(self, tmpdir):
        # print("foo")
        # print(type(tmpdir))
        self.mkdir = tmpdir.mkdir

    def test_find_dataset_dir_fallback(self) -> None:
        tmpdir = self.mkdir("test_find_dataset_dir_fallback")
        with mock.patch.dict(os.environ, {"HOME": str(tmpdir)}):
            with self.assertRaises(ValueError):
                find_cdrpy_dataset_dir()

    def test_find_dataset_dir_from_explicit_path(self) -> None:
        tmpdir = self.mkdir("test_find_dataset_dir_from_explicit_path")
        dataset_dir = tmpdir.ensure(*DEFAULT_CDRPY_DATASET_DIR.split("/"))

        assert find_cdrpy_dataset_dir(str(dataset_dir)) == str(dataset_dir)

    def test_find_config_from_environment(self) -> None:
        tmpdir = self.mkdir("test_find_datset_dir_from_environment")
        dataset_dir = tmpdir.ensure(*DEFAULT_CDRPY_DATASET_DIR.split("/"))

        with mock.patch.dict(os.environ, {CDRPY_DATA_PREFIX: str(tmpdir)}):
            assert find_cdrpy_dataset_dir() == str(dataset_dir)

    @mark.skipif("sys.platform == 'win32'")
    def test_find_dataset_dir_from_home_posix(self) -> None:
        tmpdir = self.mkdir("test_find_dataset_dir_from_home_posix")
        dataset_dir = tmpdir.ensure(*DEFAULT_CDRPY_DATASET_DIR.split("/"))

        with mock.patch.dict(os.environ, {"HOME": str(tmpdir)}):
            assert find_cdrpy_dataset_dir() == str(dataset_dir)

    @mark.skipif("sys.platform != 'win32'")
    def test_find_dataset_dir_from_home_windows(self):
        tmpdir = self.mkdir("test_find_dataset_dir_from_home_windows")
        dataset_dir = tmpdir.ensure(*DEFAULT_CDRPY_DATASET_DIR.split("/"))

        with mock.patch.dict(os.environ, {"USERPROFILE": str(tmpdir)}):
            assert find_cdrpy_dataset_dir() == str(dataset_dir)
