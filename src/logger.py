import logging
import os
import sys
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path('logs')


def get_logger(name: str) -> logging.Logger:
    """
    Get a custom logger by name.

    Args:
        name: Logger name, typically passed as __name__ from the
            calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    LOGS_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler — INFO and above only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler — DEBUG and above (everything)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f'{timestamp}.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    return logger
