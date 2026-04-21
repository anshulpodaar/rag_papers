"""Factory for creating experiment trackers."""

from src.config import config
from src.logger import get_logger
from src.tracking.base import BaseTracker

logger = get_logger(__name__)


def get_tracker(backend: str | None = None) -> BaseTracker:
    """
    Get an experiment tracker instance.

    Factory function that returns the appropriate tracker based on
    config.yaml settings. No code changes needed to switch backends.

    Args:
        backend: Override backend ('json', 'mlflow', or 'both').
            If None, uses config.yaml tracking.backend value.

    Returns:
        BaseTracker instance.

    Raises:
        ValueError: If backend is not supported.

    Examples:
        # Use config.yaml setting
        tracker = get_tracker()

        # Override config
        tracker = get_tracker('json')
    """
    tracking_config = config.get('tracking', {})

    # Determine backend
    if backend is None:
        backend = tracking_config.get('backend', 'json')

    backend = backend.lower()

    # Get backend-specific config
    json_config = tracking_config.get('json', {})
    mlflow_config = tracking_config.get('mlflow', {})

    if backend == 'json':
        from src.tracking.json_tracker import JsonTracker
        tracker = JsonTracker(
            experiments_dir=json_config.get('experiments_dir', './experiments'),
        )
        logger.info('Using JSON tracker')

    elif backend == 'mlflow':
        from src.tracking.mlflow_tracker import MlflowTracker
        tracker = MlflowTracker(
            experiment_name=mlflow_config.get('experiment_name', 'rag_papers'),
            tracking_uri=mlflow_config.get('tracking_uri'),
        )
        logger.info('Using MLflow tracker')

    elif backend == 'both':
        from src.tracking.composite_tracker import CompositeTracker
        from src.tracking.json_tracker import JsonTracker
        from src.tracking.mlflow_tracker import MlflowTracker

        trackers = [
            JsonTracker(
                experiments_dir=json_config.get('experiments_dir', './experiments'),
            ),
            MlflowTracker(
                experiment_name=mlflow_config.get('experiment_name', 'rag_papers'),
                tracking_uri=mlflow_config.get('tracking_uri'),
            ),
        ]
        tracker = CompositeTracker(trackers)
        logger.info('Using both JSON and MLflow trackers')

    else:
        raise ValueError(
            f"Unknown tracker backend: '{backend}'. "
            f"Supported: 'json', 'mlflow', 'both'"
        )

    return tracker
