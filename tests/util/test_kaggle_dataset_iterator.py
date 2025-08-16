import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import os
import json
from tqdm import tqdm
from ApexDAG.util.kaggle_dataset_iterator import KaggleDatasetIterator

class TestKaggleDatasetIterator(unittest.TestCase):

    def setUp(self):
        self.main_folder = "/mock/main_folder"
        self.iterator = KaggleDatasetIterator(self.main_folder)

    def test_initialization(self):
        self.assertEqual(self.iterator.main_folder, self.main_folder)
        self.assertEqual(self.iterator.results, [])
        self.assertIsNone(self.iterator._iterator)

    @patch.object(KaggleDatasetIterator, '_process_folders')
    @patch('tqdm.tqdm')
    def test_iter(self, mock_tqdm, mock_process_folders):
        mock_process_folders.return_value = None # _process_folders modifies self.results
        self.iterator.results = [{"item": 1}, {"item": 2}]
        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value = mock_tqdm_instance
        mock_tqdm_instance.__iter__.return_value = iter([{"item": 1}, {"item": 2}])

        results = list(self.iterator)

        mock_process_folders.assert_called_once()
        mock_tqdm.assert_called_once_with(self.iterator.results, desc="Processed competitions")
        self.assertEqual(results, [{"item": 1}, {"item": 2}])
        self.assertEqual(self.iterator._iterator, mock_tqdm_instance)

    def test_print_with_iterator(self):
        self.iterator._iterator = MagicMock()
        self.iterator.print("test message")
        self.iterator._iterator.write.assert_called_once_with("test message")

    def test_print_without_iterator(self):
        self.iterator._iterator = None
        with self.assertRaises(RuntimeError):
            self.iterator.print("test message")

    @patch('os.listdir', side_effect=[['subfolder1'], ['file1.json', 'notebook1.ipynb']])
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    @patch('os.path.isdir', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    @patch('json.load', return_value={'key': 'value'})
    def test_process_folders(self, mock_json_load, mock_open_file, mock_isdir, mock_join, mock_listdir):
        self.iterator._process_folders()

        self.assertEqual(len(self.iterator.results), 1)
        result = self.iterator.results[0]
        self.assertEqual(result['subfolder'], 'subfolder1')
        self.assertEqual(result['subfolder_path'], '/mock/main_folder/subfolder1')
        self.assertEqual(result['json_file'], {'key': 'value'})
        self.assertEqual(result['ipynb_files'], ['notebook1.ipynb'])

        mock_listdir.assert_has_calls([
            call('/mock/main_folder'),
            call('/mock/main_folder/subfolder1')
        ])
        mock_isdir.assert_called_once_with('/mock/main_folder/subfolder1')
        mock_open_file.assert_called_once_with('/mock/main_folder/subfolder1/file1.json', "r", encoding="utf-8")
        mock_json_load.assert_called_once_with(mock_open_file())

if __name__ == '__main__':
    unittest.main()
