"""Tracking-specific test fixtures."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_mlflow():
    """
    Return a fully mocked mlflow module.

    Patches the global mlflow variable in the mlflow_tracker module
    so no real MLflow operations occur.
    """
    mock = MagicMock()
    mock.start_run.return_value = MagicMock()
    mock.get_experiment_by_name.return_value = MagicMock(
        experiment_id='test_exp_id'
    )
    with patch('src.tracking.mlflow_tracker.mlflow', mock):
        yield mock


@pytest.fixture
def tmp_experiments_dir(tmp_path):
    """
    Return a temporary directory path for JsonTracker experiments.

    Automatically cleaned up by pytest after each test.
    """
    return str(tmp_path / 'experiments')
