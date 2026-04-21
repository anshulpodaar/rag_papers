"""JSON-based experiment tracking implementation."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.logger import get_logger
from src.tracking.base import BaseTracker

logger = get_logger(__name__)


class JsonTracker(BaseTracker):
    """
    File-based experiment tracking using JSON.

    Stores experiments in a directory structure:
        experiments/
        ├── index.json
        ├── exp_001/
        │   ├── config.json
        │   ├── results.json
        │   ├── summary.json
        │   └── artifacts/
        ├── exp_002/
        │   └── ...

    Simple, portable, human-readable. Good for debugging and small-scale
    experiments. No external dependencies.
    """

    def __init__(self, experiments_dir: str = './experiments') -> None:
        """
        Initialize the JSON tracker.

        Args:
            experiments_dir: Directory to store experiment data.
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)

        self._index_path = self.experiments_dir / 'index.json'
        self._ensure_index()

        self._current_experiment: str | None = None
        self._current_config: dict = {}
        self._results: list[dict] = []
        self._metrics: dict[str, list] = {}

        logger.info('JsonTracker initialised at %s', self.experiments_dir)

    def start_experiment(
        self,
        experiment_id: str,
        config: dict,
        description: str = '',
    ) -> None:
        """
        Start a new experiment run.

        Args:
            experiment_id: Unique identifier for this experiment.
            config: Dictionary of configuration parameters.
            description: Human-readable description.
        """
        self._current_experiment = experiment_id
        self._results = []
        self._metrics = {}

        exp_dir = self._get_exp_dir(experiment_id)
        exp_dir.mkdir(exist_ok=True)
        (exp_dir / 'artifacts').mkdir(exist_ok=True)

        self._current_config = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            **config,
        }

        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(self._current_config, f, indent=2)

        self._update_index(experiment_id, 'running', description)
        logger.info('Started experiment: %s', experiment_id)

    def log_result(
        self,
        question: str,
        answer: str,
        metrics: dict,
        **extra,
    ) -> None:
        """
        Log a single question-answer result.

        Args:
            question: The question that was asked.
            answer: The generated answer.
            metrics: Dictionary of metrics.
            **extra: Additional data to log.
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'metrics': metrics,
            **extra,
        }
        self._results.append(result)

        # Also track metrics for time-series
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)

        logger.debug('Logged result for: %s', question[:50])

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Log a single metric value.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        if key not in self._metrics:
            self._metrics[key] = []

        if step is not None:
            # Pad list if needed
            while len(self._metrics[key]) <= step:
                self._metrics[key].append(None)
            self._metrics[key][step] = value
        else:
            self._metrics[key].append(value)

    def log_artifact(self, filepath: str) -> None:
        """
        Copy a file to the experiment's artifacts directory.

        Args:
            filepath: Path to the file to log.
        """
        if not self._current_experiment:
            raise RuntimeError('No active experiment. Call start_experiment first.')

        src = Path(filepath)
        if not src.exists():
            logger.warning('Artifact not found: %s', filepath)
            return

        dest = self._get_exp_dir(self._current_experiment) / 'artifacts' / src.name
        shutil.copy2(src, dest)
        logger.debug('Logged artifact: %s', src.name)

    def end_experiment(self) -> dict:
        """
        End the current experiment and compute summary statistics.

        Returns:
            Dictionary containing summary metrics.
        """
        if not self._current_experiment:
            raise RuntimeError('No active experiment.')

        exp_dir = self._get_exp_dir(self._current_experiment)

        # Save detailed results
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(self._results, f, indent=2)

        # Calculate and save summary
        summary = self._calculate_summary()
        with open(exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        self._update_index(self._current_experiment, 'completed')

        logger.info(
            'Ended experiment: %s (%d results)',
            self._current_experiment,
            len(self._results),
        )

        self._current_experiment = None
        self._current_config = {}
        self._results = []
        self._metrics = {}

        return summary

    def get_experiment_summary(self, experiment_id: str) -> dict:
        """
        Retrieve summary for a past experiment.

        Args:
            experiment_id: ID of the experiment to retrieve.

        Returns:
            Dictionary containing experiment config and summary metrics.
        """
        exp_dir = self._get_exp_dir(experiment_id)

        if not exp_dir.exists():
            raise ValueError(f'Experiment not found: {experiment_id}')

        result = {}

        config_path = exp_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                result['config'] = json.load(f)

        summary_path = exp_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                result['summary'] = json.load(f)

        return result

    def list_experiments(self) -> list[dict]:
        """
        List all experiments.

        Returns:
            List of experiment summaries.
        """
        with open(self._index_path) as f:
            index = json.load(f)

        return index.get('experiments', [])

    def compare_experiments(self, experiment_ids: list[str]) -> dict:
        """
        Compare multiple experiments side by side.

        Args:
            experiment_ids: List of experiment IDs to compare.

        Returns:
            Dictionary with comparison data.
        """
        comparison = {}

        for exp_id in experiment_ids:
            try:
                data = self.get_experiment_summary(exp_id)
                comparison[exp_id] = {
                    'description': data.get('config', {}).get('description', ''),
                    'metrics': data.get('summary', {}).get('metrics', {}),
                }
            except ValueError:
                comparison[exp_id] = {'error': 'Not found'}

        return comparison

    # ── Private helpers ───────────────────────────────────────────────

    def _get_exp_dir(self, experiment_id: str) -> Path:
        """Get the directory path for an experiment."""
        return self.experiments_dir / experiment_id

    def _ensure_index(self) -> None:
        """Create index file if it doesn't exist."""
        if not self._index_path.exists():
            with open(self._index_path, 'w') as f:
                json.dump({'experiments': []}, f, indent=2)

    def _update_index(
        self,
        experiment_id: str,
        status: str,
        description: str = '',
    ) -> None:
        """Update the experiments index."""
        with open(self._index_path) as f:
            index = json.load(f)

        # Find or create entry
        experiments = index.get('experiments', [])
        entry = next((e for e in experiments if e['id'] == experiment_id), None)

        if entry:
            entry['status'] = status
            entry['updated_at'] = datetime.now().isoformat()
        else:
            experiments.append({
                'id': experiment_id,
                'status': status,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            })

        index['experiments'] = experiments

        with open(self._index_path, 'w') as f:
            json.dump(index, f, indent=2)

    def _calculate_summary(self) -> dict:
        """Calculate summary statistics from results."""
        summary = {
            'experiment_id': self._current_experiment,
            'total_questions': len(self._results),
            'completed_at': datetime.now().isoformat(),
            'metrics': {},
        }

        if not self._results:
            return summary

        # Aggregate metrics
        for key, values in self._metrics.items():
            clean_values = [v for v in values if v is not None]
            if clean_values:
                summary['metrics'][key] = {
                    'mean': sum(clean_values) / len(clean_values),
                    'min': min(clean_values),
                    'max': max(clean_values),
                    'count': len(clean_values),
                }

        return summary
