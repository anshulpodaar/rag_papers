"""Tests for the CompositeTracker that forwards to multiple backends."""

from unittest.mock import MagicMock

import pytest

from src.tracking.base import BaseTracker
from src.tracking.composite_tracker import CompositeTracker


@pytest.fixture
def mock_trackers():
    """Return a list of two mock trackers with spec=BaseTracker."""
    tracker_a = MagicMock(spec=BaseTracker)
    tracker_b = MagicMock(spec=BaseTracker)
    tracker_a.end_experiment.return_value = {'total_questions': 3, 'metrics': {}}
    tracker_b.end_experiment.return_value = {'total_questions': 3, 'metrics': {}}
    return [tracker_a, tracker_b]


class TestCompositeTrackerInit:
    """Tests for CompositeTracker instantiation."""

    def test_empty_tracker_list_raises_value_error(self):
        """Instantiation with an empty list raises ValueError."""
        with pytest.raises(ValueError, match='At least one tracker required'):
            CompositeTracker([])


class TestCompositeTrackerStartExperiment:
    """Tests for start_experiment() forwarding."""

    def test_forwards_to_all_child_trackers(self, mock_trackers):
        """start_experiment() is called on each child tracker."""
        composite = CompositeTracker(mock_trackers)
        config = {'model': 'test'}

        composite.start_experiment('exp_001', config, 'description')

        for tracker in mock_trackers:
            tracker.start_experiment.assert_called_once_with(
                'exp_001', config, 'description'
            )


class TestCompositeTrackerLogResult:
    """Tests for log_result() forwarding."""

    def test_forwards_to_all_child_trackers(self, mock_trackers):
        """log_result() is called on each child tracker."""
        composite = CompositeTracker(mock_trackers)

        composite.log_result('What is AI?', 'AI is...', {'score': 0.9})

        for tracker in mock_trackers:
            tracker.log_result.assert_called_once_with(
                'What is AI?', 'AI is...', {'score': 0.9}
            )


class TestCompositeTrackerLogMetric:
    """Tests for log_metric() forwarding."""

    def test_forwards_to_all_child_trackers(self, mock_trackers):
        """log_metric() is called on each child tracker."""
        composite = CompositeTracker(mock_trackers)

        composite.log_metric('accuracy', 0.95, step=1)

        for tracker in mock_trackers:
            tracker.log_metric.assert_called_once_with('accuracy', 0.95, 1)


class TestCompositeTrackerEndExperiment:
    """Tests for end_experiment() forwarding."""

    def test_forwards_to_all_and_returns_first_summary(self, mock_trackers):
        """end_experiment() calls all child trackers and returns first summary."""
        composite = CompositeTracker(mock_trackers)

        summary = composite.end_experiment()

        for tracker in mock_trackers:
            tracker.end_experiment.assert_called_once()

        # Returns the first tracker's summary
        assert summary == {'total_questions': 3, 'metrics': {}}
