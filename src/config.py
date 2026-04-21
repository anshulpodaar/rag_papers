"""Configuration loader for the RAG pipeline."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

CONFIG_PATH = Path('config.yaml')


def load_config() -> dict:
    """
    Load configuration from the YAML file.

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


def get_api_key(service: str) -> str | None:
    """
    Get API key for a service from environment variables.

    Args:
        service: Service name ('anthropic', 'openai', 'huggingface').

    Returns:
        API key string or None if not set.
    """
    key_map = {
        'anthropic': 'ANTHROPIC_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'huggingface': 'HF_TOKEN',
    }
    env_var = key_map.get(service.lower())
    if not env_var:
        raise ValueError(f'Unknown service: {service}')

    return os.getenv(env_var)


config = load_config()
