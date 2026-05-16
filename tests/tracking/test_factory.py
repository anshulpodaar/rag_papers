"""Tests for the tracker factory function."""

from unittest.mock import MagicMock, patch

import pytest

from src.tracking.composite_tracker import CompositeTracker
from src.tracking.factory import get_tracker
from src.tracking.json_tracker import JsonTracker
from src.tracking.mlflow_tracker import MlflowTracker


class TestGetTrackerJson:
    """Tests for get_tracker('json')."""

    def test_returns_json_tracker(self, tmp_path):
        """get_tracker('json') returns a JsonTracker instance."""
        with patch('src.tracking.factory.config', {
            'tracking': {
                'backend': 'json',
                'json': {'experiments_dir': str(tmp_path / 'experiments')},
                'mlflow': {},
            },
        }):
            tracker = get_tracker('json')
            assert isinstance(tracker, JsonTracker)


class TestGetTrackerMlflow:
    """Tests for get_tracker('mlflow')."""

    def test_returns_mlflow_tracker(self, mock_mlflow):
        """get_tracker('mlflow') returns an MlflowTracker instance."""
        with patch('src.tracking.factory.config', {
            'tracking': {
                'backend': 'json',
                'json': {},
                'mlflow': {
                    'experiment_name': 'test_rag',
                    'tracking_uri': None,
                },
            },
        }):
            tracker = get_tracker('mlflow')
            assert isinstance(tracker, MlflowTracker)


class TestGetTrackerBoth:
    """Tests for get_tracker('both')."""

    def test_returns_composite_tracker(self, tmp_path, mock_mlflow):
        """get_tracker('both') returns a CompositeTracker instance."""
        with patch('src.tracking.factory.config', {
            'tracking': {
                'backend': 'json',
                'json': {'experiments_dir': str(tmp_path / 'experiments')},
                'mlflow': {
                    'experiment_name': 'test_rag',
                    'tracking_uri': None,
                },
            },
        }):
            tracker = get_tracker('both')
            assert isinstance(tracker, CompositeTracker)


class TestGetTrackerUnknown:
    """Tests for unknown backend."""

    def test_raises_value_error(self):
        """get_tracker() with unknown backend raises ValueError."""
        with patch('src.tracking.factory.config', {
            'tracking': {
                'backend': 'json',
                'json': {},
                'mlflow': {},
            },
        }):
            with pytest.raises(ValueError, match='Unknown tracker backend'):
                get_tracker('unknown_backend')


class TestGetTrackerNoneFallback:
    """Tests for None backend falling back to config."""

    def test_falls_back_to_config_value(self, tmp_path):
        """get_tracker(None) falls back to the config tracking.backend value."""
        with patch('src.tracking.factory.config', {
            'tracking': {
                'backend': 'json',
                'json': {'experiments_dir': str(tmp_path / 'experiments')},
                'mlflow': {},
            },
        }):
            tracker = get_tracker(None)
            assert isinstance(tracker, JsonTracker)
