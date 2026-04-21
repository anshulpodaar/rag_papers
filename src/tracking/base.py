"""Abstract base class for experiment tracking."""

from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """
    Abstract interface for experiment tracking.

    All tracker implementations must inherit from this class and implement
    the required methods. This allows swapping between JSON, MLflow, or
    other backends without changing experiment code.
    """

    @abstractmethod
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
            config: Dictionary of configuration parameters (chunking, LLM, etc.).
            description: Human-readable description of what's being tested.
        """
        pass

    @abstractmethod
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
            metrics: Dictionary of metrics (faithfulness, latency, tokens, etc.).
            **extra: Additional data to log (chunks retrieved, sources, etc.).
        """
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Log a single metric value.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number for time-series metrics.
        """
        pass

    @abstractmethod
    def log_artifact(self, filepath: str) -> None:
        """
        Log a file as an artifact.

        Args:
            filepath: Path to the file to log.
        """
        pass

    @abstractmethod
    def end_experiment(self) -> dict:
        """
        End the current experiment and compute summary statistics.

        Returns:
            Dictionary containing summary metrics.
        """
        pass

    @abstractmethod
    def get_experiment_summary(self, experiment_id: str) -> dict:
        """
        Retrieve summary for a past experiment.

        Args:
            experiment_id: ID of the experiment to retrieve.

        Returns:
            Dictionary containing experiment config and summary metrics.
        """
        pass

    @abstractmethod
    def list_experiments(self) -> list[dict]:
        """
        List all experiments.

        Returns:
            List of experiment summaries.
        """
        pass
