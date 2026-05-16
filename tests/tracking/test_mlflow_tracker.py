"""Tests for the MlflowTracker experiment tracking implementation."""

from unittest.mock import MagicMock, patch

import pytest

from src.tracking.mlflow_tracker import MlflowTracker, _ensure_mlflow


class TestMlflowTrackerInit:
    """Tests for MlflowTracker instantiation."""

    def test_calls_set_experiment(self, mock_mlflow):
        """Instantiation calls mlflow.set_experiment() with the experiment name."""
        tracker = MlflowTracker(experiment_name='test_experiment')

        mock_mlflow.set_experiment.assert_called_once_with('test_experiment')


class TestMlflowTrackerStartExperiment:
    """Tests for start_experiment()."""

    def test_calls_start_run_and_logs_params(self, mock_mlflow):
        """start_experiment() calls mlflow.start_run() and logs parameters."""
        tracker = MlflowTracker(experiment_name='test_exp')

        config = {'chunk_size': 200, 'model': 'test-model'}
        tracker.start_experiment('run_001', config, 'Test run')

        mock_mlflow.start_run.assert_called_once_with(run_name='run_001')
        mock_mlflow.set_tag.assert_any_call('description', 'Test run')
        # Verify params are logged (flattened)
        mock_mlflow.log_param.assert_any_call('chunk_size', '200')
        mock_mlflow.log_param.assert_any_call('model', 'test-model')


class TestMlflowTrackerLogResult:
    """Tests for log_result()."""

    def test_calls_log_metric_for_each_metric(self, mock_mlflow):
        """log_result() calls mlflow.log_metric() for each numeric metric."""
        tracker = MlflowTracker(experiment_name='test_exp')
        tracker.start_experiment('run_001', {})

        tracker.log_result('What is AI?', 'AI is...', {'score': 0.9, 'latency': 1.2})

        mock_mlflow.log_metric.assert_any_call('score', 0.9, step=0)
        mock_mlflow.log_metric.assert_any_call('latency', 1.2, step=0)


class TestMlflowTrackerEndExperiment:
    """Tests for end_experiment()."""

    def test_calls_end_run(self, mock_mlflow):
        """end_experiment() calls mlflow.end_run()."""
        tracker = MlflowTracker(experiment_name='test_exp')
        tracker.start_experiment('run_001', {})
        tracker.log_result('Q1', 'A1', {'score': 0.9})

        tracker.end_experiment()

        mock_mlflow.end_run.assert_called_once()


class TestEnsureMlflow:
    """Tests for _ensure_mlflow() lazy import."""

    def test_raises_import_error_when_not_installed(self):
        """_ensure_mlflow() raises ImportError when mlflow is not installed."""
        import src.tracking.mlflow_tracker as module

        # Reset the global mlflow to None to force re-import
        original = module.mlflow
        module.mlflow = None

        try:
            with patch.dict('sys.modules', {'mlflow': None}):
                with patch('builtins.__import__', side_effect=ImportError('No module named mlflow')):
                    with pytest.raises(ImportError, match='MLflow is not installed'):
                        _ensure_mlflow()
        finally:
            # Restore original state
            module.mlflow = original
