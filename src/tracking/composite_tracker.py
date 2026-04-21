"""Composite tracker that logs to multiple backends simultaneously."""

from src.logger import get_logger
from src.tracking.base import BaseTracker

logger = get_logger(__name__)


class CompositeTracker(BaseTracker):
    """
    Logs to multiple tracking backends simultaneously.

    Useful for running JSON (for debugging/portability) and MLflow
    (for UI/comparison) in parallel during experiments.

    All method calls are forwarded to each underlying tracker.
    Results are aggregated where applicable.
    """

    def __init__(self, trackers: list[BaseTracker]) -> None:
        """
        Initialize with multiple trackers.

        Args:
            trackers: List of tracker instances to log to.
        """
        if not trackers:
            raise ValueError('At least one tracker required')

        self._trackers = trackers
        tracker_names = [t.__class__.__name__ for t in trackers]
        logger.info('CompositeTracker initialised with: %s', tracker_names)

    def start_experiment(
        self,
        experiment_id: str,
        config: dict,
        description: str = '',
    ) -> None:
        """Start experiment on all trackers."""
        for tracker in self._trackers:
            tracker.start_experiment(experiment_id, config, description)

    def log_result(
        self,
        question: str,
        answer: str,
        metrics: dict,
        **extra,
    ) -> None:
        """Log result to all trackers."""
        for tracker in self._trackers:
            tracker.log_result(question, answer, metrics, **extra)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log metric to all trackers."""
        for tracker in self._trackers:
            tracker.log_metric(key, value, step)

    def log_artifact(self, filepath: str) -> None:
        """Log artifact to all trackers."""
        for tracker in self._trackers:
            tracker.log_artifact(filepath)

    def end_experiment(self) -> dict:
        """
        End experiment on all trackers.

        Returns:
            Summary from the first tracker (they should be identical).
        """
        summaries = []
        for tracker in self._trackers:
            summaries.append(tracker.end_experiment())

        # Return first summary (they should match)
        return summaries[0] if summaries else {}

    def get_experiment_summary(self, experiment_id: str) -> dict:
        """
        Get summary from the first tracker.

        Args:
            experiment_id: ID of the experiment.

        Returns:
            Summary from the first tracker.
        """
        return self._trackers[0].get_experiment_summary(experiment_id)

    def list_experiments(self) -> list[dict]:
        """
        List experiments from the first tracker.

        Returns:
            List of experiments from the first tracker.
        """
        return self._trackers[0].list_experiments()
