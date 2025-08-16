import unittest
from unittest.mock import MagicMock, patch, call
import logging
import os
from ApexDAG.util.logging import setup_wandb, setup_logging

class TestLogging(unittest.TestCase):

    @patch('wandb.init')
    @patch('os.getenv', return_value='test_user')
    @patch('ApexDAG.util.logging.load_dotenv')
    def test_setup_wandb_no_name(self, mock_load_dotenv, mock_getenv, mock_wandb_init):
        setup_wandb("test_project")
        mock_wandb_init.assert_called_once_with(project="test_project", entity="test_user")
        mock_getenv.assert_called_once_with("WANDB_USER", "default_user")

    @patch('wandb.init')
    @patch('os.getenv', return_value='test_user')
    @patch('ApexDAG.util.logging.load_dotenv')
    def test_setup_wandb_with_name(self, mock_load_dotenv, mock_getenv, mock_wandb_init):
        setup_wandb("test_project", name="my-run-name")
        mock_wandb_init.assert_called_once_with(project="test_project", entity="test_user", name="hash_my_run_name")

    @patch('logging.StreamHandler')
    @patch('logging.Formatter')
    @patch('logging.getLogger')
    def test_setup_logging_verbose(self, mock_get_logger, mock_formatter, mock_stream_handler):
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.handlers = [] # Simulate no existing handlers
        mock_get_logger.return_value = mock_logger

        logger = setup_logging("test_logger", verbose=True)

        mock_get_logger.assert_called_once_with("test_logger")
        mock_logger.setLevel.assert_called_once_with(logging.INFO)
        mock_stream_handler.assert_called_once()
        mock_formatter.assert_called_once_with('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        mock_stream_handler.return_value.setFormatter.assert_called_once_with(mock_formatter.return_value)
        mock_logger.addHandler.assert_called_once_with(mock_stream_handler.return_value)
        self.assertEqual(logger, mock_logger)

    @patch('logging.StreamHandler')
    @patch('logging.Formatter')
    @patch('logging.getLogger')
    def test_setup_logging_not_verbose(self, mock_get_logger, mock_formatter, mock_stream_handler):
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        logger = setup_logging("test_logger", verbose=False)

        mock_logger.setLevel.assert_called_once_with(logging.ERROR)

    @patch('logging.StreamHandler')
    @patch('logging.Formatter')
    @patch('logging.getLogger')
    def test_setup_logging_existing_handlers(self, mock_get_logger, mock_formatter, mock_stream_handler):
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.handlers = [MagicMock()] # Simulate existing handlers
        mock_get_logger.return_value = mock_logger

        setup_logging("test_logger", verbose=True)

        mock_logger.addHandler.assert_not_called()

if __name__ == '__main__':
    unittest.main()
