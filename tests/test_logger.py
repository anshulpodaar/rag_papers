"""Unit tests for src.logger module.

Tests dual-handler setup (console + file), log levels, file creation,
and duplicate handler prevention.
"""

import logging
from pathlib import Path

from src.logger import get_logger

# ── Constants ─────────────────────────────────────────────────────────────────

EXPECTED_HANDLER_COUNT = 2
CONSOLE_HANDLER_LEVEL = logging.INFO
FILE_HANDLER_LEVEL = logging.DEBUG
LOGGER_LEVEL = logging.DEBUG


# ── Tests: Handler Setup ──────────────────────────────────────────────────────


class TestGetLoggerHandlers:
    """Tests for get_logger() handler configuration."""

    def test_returns_logger_with_two_handlers(self, tmp_path: Path, monkeypatch) -> None:
        """Verify get_logger returns a Logger with both console and file handlers."""
        monkeypatch.setattr('src.logger.LOGS_DIR', tmp_path)

        logger = get_logger('test.handlers')

        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == EXPECTED_HANDLER_COUNT

    def test_console_handler_is_stream_handler_at_info(self, tmp_path: Path, monkeypatch) -> None:
        """Verify the console handler is a StreamHandler set to INFO level."""
        monkeypatch.setattr('src.logger.LOGS_DIR', tmp_path)

        logger = get_logger('test.console_level')
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]

        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == CONSOLE_HANDLER_LEVEL

    def test_file_handler_is_file_handler_at_debug(self, tmp_path: Path, monkeypatch) -> None:
        """Verify the file handler is a FileHandler set to DEBUG level."""
        monkeypatch.setattr('src.logger.LOGS_DIR', tmp_path)

        logger = get_logger('test.file_level')
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]

        assert len(file_handlers) == 1
        assert file_handlers[0].level == FILE_HANDLER_LEVEL

    def test_logger_level_is_debug(self, tmp_path: Path, monkeypatch) -> None:
        """Verify the logger itself is set to DEBUG level."""
        monkeypatch.setattr('src.logger.LOGS_DIR', tmp_path)

        logger = get_logger('test.logger_level')

        assert logger.level == LOGGER_LEVEL


# ── Tests: Log File Creation ──────────────────────────────────────────────────


class TestLogFileCreation:
    """Tests for log file creation on disk."""

    def test_log_file_created_in_logs_dir(self, tmp_path: Path, monkeypatch) -> None:
        """Verify get_logger creates a .log file inside the configured LOGS_DIR."""
        monkeypatch.setattr('src.logger.LOGS_DIR', tmp_path)

        get_logger('test.file_creation')

        log_files = list(tmp_path.glob('*.log'))
        assert len(log_files) == 1

    def test_logs_directory_created_if_missing(self, tmp_path: Path, monkeypatch) -> None:
        """Verify get_logger creates the LOGS_DIR directory if it does not exist."""
        logs_dir = tmp_path / 'new_logs'
        monkeypatch.setattr('src.logger.LOGS_DIR', logs_dir)

        get_logger('test.dir_creation')

        assert logs_dir.exists()
        assert logs_dir.is_dir()


# ── Tests: Duplicate Handler Prevention ───────────────────────────────────────


class TestDuplicateHandlerPrevention:
    """Tests for duplicate handler prevention on repeated calls."""

    def test_repeated_calls_return_same_instance(self, tmp_path: Path, monkeypatch) -> None:
        """Verify calling get_logger twice with same name returns the same logger."""
        monkeypatch.setattr('src.logger.LOGS_DIR', tmp_path)

        logger_first = get_logger('test.duplicate')
        logger_second = get_logger('test.duplicate')

        assert logger_first is logger_second

    def test_repeated_calls_do_not_add_duplicate_handlers(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Verify calling get_logger twice does not double the handler count."""
        monkeypatch.setattr('src.logger.LOGS_DIR', tmp_path)

        get_logger('test.no_dupes')
        logger = get_logger('test.no_dupes')

        assert len(logger.handlers) == EXPECTED_HANDLER_COUNT
