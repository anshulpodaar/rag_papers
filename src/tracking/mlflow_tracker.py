"""MLflow-based experiment tracking implementation."""

from datetime import datetime

from src.logger import get_logger
from src.tracking.base import BaseTracker

logger = get_logger(__name__)

# Lazy import to avoid dependency if not used
mlflow = None


def _ensure_mlflow():
    """Lazy load MLflow to avoid import errors if not installed."""
    global mlflow
    if mlflow is None:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
        except ImportError:
            raise ImportError(
                'MLflow is not installed. Run: pip install mlflow'
            )


class MlflowTracker(BaseTracker):
    """
    MLflow-based experiment tracking.

    Provides a web UI for visualizing experiments, comparing runs,
    and storing artifacts. Industry standard for ML experiment tracking.

    Start the UI with: mlflow ui
    Then open: http://localhost:5000
    """

    def __init__(
        self,
        experiment_name: str = 'rag_papers',
        tracking_uri: str | None = None,
    ) -> None:
        """
        Initialize the MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: MLflow tracking server URI. Defaults to local ./mlruns.
        """
        _ensure_mlflow()

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        self._experiment_name = experiment_name
        self._run = None
        self._current_experiment: str | None = None
        self._results: list[dict] = []
        self._step = 0

        logger.info('MlflowTracker initialised for experiment: %s', experiment_name)

    def start_experiment(
        self,
        experiment_id: str,
        config: dict,
        description: str = '',
    ) -> None:
        """
        Start a new MLflow run.

        Args:
            experiment_id: Used as the run name.
            config: Dictionary of configuration parameters.
            description: Human-readable description.
        """
        self._current_experiment = experiment_id
        self._results = []
        self._step = 0

        self._run = mlflow.start_run(run_name=experiment_id)

        # Log description as tag
        mlflow.set_tag('description', description)
        mlflow.set_tag('timestamp', datetime.now().isoformat())

        # Flatten and log config as params
        self._log_params_recursive(config)

        logger.info('Started MLflow run: %s', experiment_id)

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
        # Store for summary calculation
        self._results.append({
            'question': question,
            'answer': answer,
            'metrics': metrics,
            **extra,
        })

        # Log each metric with step
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=self._step)

        self._step += 1
        logger.debug('Logged result %d for: %s', self._step, question[:50])

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Log a single metric value.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        mlflow.log_metric(key, value, step=step or self._step)

    def log_artifact(self, filepath: str) -> None:
        """
        Log a file as an MLflow artifact.

        Args:
            filepath: Path to the file to log.
        """
        mlflow.log_artifact(filepath)
        logger.debug('Logged artifact: %s', filepath)

    def end_experiment(self) -> dict:
        """
        End the current MLflow run and log summary metrics.

        Returns:
            Dictionary containing summary metrics.
        """
        if not self._run:
            raise RuntimeError('No active run. Call start_experiment first.')

        summary = self._calculate_summary()

        # Log summary metrics with 'avg_' prefix
        for key, stats in summary.get('metrics', {}).items():
            mlflow.log_metric(f'avg_{key}', stats['mean'])
            mlflow.log_metric(f'min_{key}', stats['min'])
            mlflow.log_metric(f'max_{key}', stats['max'])

        mlflow.log_metric('total_questions', summary['total_questions'])

        mlflow.end_run()

        logger.info(
            'Ended MLflow run: %s (%d results)',
            self._current_experiment,
            len(self._results),
        )

        self._run = None
        self._current_experiment = None
        self._results = []
        self._step = 0

        return summary

    def get_experiment_summary(self, experiment_id: str) -> dict:
        """
        Retrieve summary for a past run.

        Args:
            experiment_id: Run name to retrieve.

        Returns:
            Dictionary containing experiment config and summary metrics.
        """
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if not experiment:
            raise ValueError(f'Experiment not found: {self._experiment_name}')

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{experiment_id}'",
        )

        if runs.empty:
            raise ValueError(f'Run not found: {experiment_id}')

        run = runs.iloc[0]

        # Extract params and metrics
        params = {k.replace('params.', ''): v for k, v in run.items() if k.startswith('params.')}
        metrics = {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}

        return {
            'config': params,
            'summary': {
                'metrics': metrics,
                'run_id': run['run_id'],
                'status': run['status'],
            },
        }

    def list_experiments(self) -> list[dict]:
        """
        List all runs in the experiment.

        Returns:
            List of run summaries.
        """
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if not experiment:
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=['start_time DESC'],
        )

        result = []
        for _, run in runs.iterrows():
            result.append({
                'id': run.get('tags.mlflow.runName', run['run_id']),
                'run_id': run['run_id'],
                'status': run['status'],
                'start_time': str(run['start_time']),
                'description': run.get('tags.description', ''),
            })

        return result

    def compare_experiments(self, experiment_ids: list[str]) -> dict:
        """
        Compare multiple runs side by side.

        Args:
            experiment_ids: List of run names to compare.

        Returns:
            Dictionary with comparison data.
        """
        comparison = {}

        for exp_id in experiment_ids:
            try:
                data = self.get_experiment_summary(exp_id)
                comparison[exp_id] = {
                    'metrics': data.get('summary', {}).get('metrics', {}),
                }
            except ValueError:
                comparison[exp_id] = {'error': 'Not found'}

        return comparison

    # ── Private helpers ───────────────────────────────────────────────

    def _log_params_recursive(self, config: dict, prefix: str = '') -> None:
        """Flatten nested config and log as MLflow params."""
        for key, value in config.items():
            param_name = f'{prefix}{key}' if prefix else key

            if isinstance(value, dict):
                self._log_params_recursive(value, f'{param_name}.')
            else:
                # MLflow params must be strings
                mlflow.log_param(param_name, str(value))

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
        all_metrics: dict[str, list] = {}
        for result in self._results:
            for key, value in result.get('metrics', {}).items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        for key, values in all_metrics.items():
            if values:
                summary['metrics'][key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                }

        return summary
