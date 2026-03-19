from pathlib import Path

import yaml

from src.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path('config.yaml')


def load_config() -> dict:
	"""
	Load configuration from the YAML file.

	Args:
		None

	Returns:
		Dict containing all configuration values.

	Raises:
		FileNotFoundError: If config.yaml does not exist.
	"""
	if not CONFIG_PATH.exists():
		raise FileNotFoundError(
				f'Config file not found at {CONFIG_PATH}. '
				f'Make sure config.yaml exists in the project root.'
		)

	with open(CONFIG_PATH, 'r') as f:
		config = yaml.safe_load(f)

	logger.debug('Config loaded from %s', CONFIG_PATH)
	return config


config = load_config()
