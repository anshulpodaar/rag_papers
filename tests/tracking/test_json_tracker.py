"""Tests for the JsonTracker experiment tracking implementation."""

import json

import pytest

from src.tracking.json_tracker import JsonTracker


class TestJsonTrackerInit:
    """Tests for JsonTracker instantiation."""

    def test_creates_experiments_directory(self, tmp_experiments_dir):
        """JsonTracker creates the experiments directory if it does not exist."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        assert tracker.experiments_dir.exists()
        assert tracker.experiments_dir.is_dir()

    def test_creates_index_file(self, tmp_experiments_dir):
        """JsonTracker creates an index.json file on init."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        index_path = tracker.experiments_dir / 'index.json'
        assert index_path.exists()
        with open(index_path) as f:
            data = json.load(f)
        assert 'experiments' in data


class TestJsonTrackerStartExperiment:
    """Tests for start_experiment()."""

    def test_initialises_state(self, tmp_experiments_dir):
        """start_experiment() initialises state with ID, parameters, and description."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        config = {'chunk_size': 200, 'model': 'test-model'}
        description = 'Test experiment'

        tracker.start_experiment('exp_001', config, description)

        assert tracker._current_experiment == 'exp_001'
        assert tracker._current_config['experiment_id'] == 'exp_001'
        assert tracker._current_config['description'] == description
        assert tracker._current_config['chunk_size'] == 200
        assert tracker._current_config['model'] == 'test-model'

    def test_creates_experiment_directory(self, tmp_experiments_dir):
        """start_experiment() creates a subdirectory for the experiment."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        tracker.start_experiment('exp_001', {'key': 'value'})

        exp_dir = tracker.experiments_dir / 'exp_001'
        assert exp_dir.exists()
        assert (exp_dir / 'config.json').exists()


class TestJsonTrackerLogResult:
    """Tests for log_result()."""

    def test_appends_to_results(self, tmp_experiments_dir):
        """log_result() appends the result to the current experiment's results."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        tracker.start_experiment('exp_001', {})

        tracker.log_result('What is AI?', 'AI is...', {'score': 0.9})
        tracker.log_result('What is ML?', 'ML is...', {'score': 0.8})

        assert len(tracker._results) == 2
        assert tracker._results[0]['question'] == 'What is AI?'
        assert tracker._results[0]['answer'] == 'AI is...'
        assert tracker._results[0]['metrics'] == {'score': 0.9}
        assert tracker._results[1]['question'] == 'What is ML?'


class TestJsonTrackerLogMetric:
    """Tests for log_metric()."""

    def test_records_metric_key_and_value(self, tmp_experiments_dir):
        """log_metric() records the metric key and value."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        tracker.start_experiment('exp_001', {})

        tracker.log_metric('accuracy', 0.95)

        assert 'accuracy' in tracker._metrics
        assert 0.95 in tracker._metrics['accuracy']

    def test_records_metric_with_step(self, tmp_experiments_dir):
        """log_metric() records the metric at the specified step."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        tracker.start_experiment('exp_001', {})

        tracker.log_metric('loss', 0.5, step=0)
        tracker.log_metric('loss', 0.3, step=1)

        assert tracker._metrics['loss'][0] == 0.5
        assert tracker._metrics['loss'][1] == 0.3


class TestJsonTrackerEndExperiment:
    """Tests for end_experiment()."""

    def test_returns_summary_with_correct_total_questions(self, tmp_experiments_dir):
        """end_experiment() returns summary with correct total_questions count."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        tracker.start_experiment('exp_001', {})
        tracker.log_result('Q1', 'A1', {'score': 0.9})
        tracker.log_result('Q2', 'A2', {'score': 0.8})

        summary = tracker.end_experiment()

        assert summary['total_questions'] == 2
        assert 'metrics' in summary
        assert 'score' in summary['metrics']
        assert summary['metrics']['score']['mean'] == pytest.approx(0.85)

    def test_raises_runtime_error_without_start(self, tmp_experiments_dir):
        """end_experiment() without start_experiment() raises RuntimeError."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)

        with pytest.raises(RuntimeError):
            tracker.end_experiment()

    def test_persists_experiment_as_json(self, tmp_experiments_dir):
        """Experiment data is persisted as JSON files."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)
        tracker.start_experiment('exp_001', {'model': 'test'})
        tracker.log_result('Q1', 'A1', {'score': 0.9})
        tracker.end_experiment()

        exp_dir = tracker.experiments_dir / 'exp_001'
        assert (exp_dir / 'results.json').exists()
        assert (exp_dir / 'summary.json').exists()

        with open(exp_dir / 'results.json') as f:
            results = json.load(f)
        assert len(results) == 1
        assert results[0]['question'] == 'Q1'

    def test_experiment_index_tracks_completed(self, tmp_experiments_dir):
        """Experiment index tracks all completed experiments."""
        tracker = JsonTracker(experiments_dir=tmp_experiments_dir)

        tracker.start_experiment('exp_001', {})
        tracker.end_experiment()

        tracker.start_experiment('exp_002', {})
        tracker.end_experiment()

        with open(tracker.experiments_dir / 'index.json') as f:
            index = json.load(f)

        experiments = index['experiments']
        assert len(experiments) == 2
        assert experiments[0]['id'] == 'exp_001'
        assert experiments[0]['status'] == 'completed'
        assert experiments[1]['id'] == 'exp_002'
        assert experiments[1]['status'] == 'completed'
