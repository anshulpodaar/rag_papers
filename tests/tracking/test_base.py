"""Tests for the BaseTracker abstract base class."""

import pytest

from src.tracking.base import BaseTracker


class TestBaseTrackerAbstract:
    """Tests verifying BaseTracker enforces the abstract interface contract."""

    def test_direct_instantiation_raises_type_error(self):
        """BaseTracker cannot be instantiated directly because it is abstract."""
        with pytest.raises(TypeError):
            BaseTracker()

    def test_abstract_methods_defined(self):
        """BaseTracker defines the required abstract methods."""
        abstract_methods = BaseTracker.__abstractmethods__
        assert 'start_experiment' in abstract_methods
        assert 'log_result' in abstract_methods
        assert 'log_metric' in abstract_methods
        assert 'log_artifact' in abstract_methods

    def test_concrete_subclass_can_be_instantiated(self):
        """A subclass implementing all abstract methods can be instantiated."""

        class ConcreteTracker(BaseTracker):
            """Minimal concrete implementation for testing."""

            def start_experiment(self, experiment_id, config, description=''):
                pass

            def log_result(self, question, answer, metrics, **extra):
                pass

            def log_metric(self, key, value, step=None):
                pass

            def log_artifact(self, filepath):
                pass

            def end_experiment(self):
                return {}

            def get_experiment_summary(self, experiment_id):
                return {}

            def list_experiments(self):
                return []

        tracker = ConcreteTracker()
        assert isinstance(tracker, BaseTracker)
