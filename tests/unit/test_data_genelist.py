import unittest

from unittest.mock import patch

from cdrpy.data.genelist import (
    load_genelist,
    GenelistEnum,
    GENE_DATA_MODULE,
    GENELIST_FILES,
)


class TestLoadGenelist(unittest.TestCase):
    @patch("cdrpy.data.genelist.load_pickled_data_resource")
    def test_load_genelist(self, mock_load_pickled_data_resource):
        mock_load_pickled_data_resource.return_value = ["gene1", "gene2", "gene3"]
        for genelist in GenelistEnum:
            result = load_genelist(genelist)
            mock_load_pickled_data_resource.assert_called_with(
                GENELIST_FILES[genelist], data_module=GENE_DATA_MODULE
            )
            self.assertEqual(result, ["gene1", "gene2", "gene3"])

    @patch("cdrpy.data.genelist.load_pickled_data_resource")
    def test_load_genelist_invalid(self, mock_load_pickled_data_resource):
        with self.assertRaises(KeyError):
            load_genelist("invalid")
