"""Unit tests for src.config module.

Tests cover:
- load_config() with valid YAML and missing file scenarios
- get_api_key() for known services, unknown services, and unset env vars

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
"""

from pathlib import Path

import pytest
import yaml

from src.config import get_api_key, load_config


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_CONFIG_DATA: dict = {
    'papers': {'path': './papers', 'filetypes': ['.pdf']},
    'vector_store': {'db_path': './db', 'collection_name': 'papers', 'n_results': 5},
    'chunking': {'chunk_size': 500, 'chunk_overlap': 50},
    'embedding': {'model_name': 'all-MiniLM-L6-v2'},
    'claude': {'model': 'claude-sonnet-4-20250514', 'max_tokens': 1024},
    'tracking': {'backend': 'json'},
}

ANTHROPIC_KEY_VALUE = 'sk-ant-test-key-12345'
OPENAI_KEY_VALUE = 'sk-openai-test-key-67890'
HUGGINGFACE_TOKEN_VALUE = 'hf_test_token_abcdef'

UNKNOWN_SERVICE_NAME = 'unsupported_service'


# ── load_config() Tests ───────────────────────────────────────────────────────


class TestLoadConfig:
    """Tests for load_config() function."""

    def test_load_config_returns_parsed_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify load_config() returns parsed dict matching YAML file contents.

        Validates: Requirements 3.1, 3.8
        """
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml.dump(VALID_CONFIG_DATA))
        monkeypatch.setattr('src.config.CONFIG_PATH', config_file)

        result = load_config()

        assert result == VALID_CONFIG_DATA

    def test_load_config_raises_file_not_found_for_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify load_config() raises FileNotFoundError when config file is missing.

        Validates: Requirement 3.2
        """
        nonexistent_path = tmp_path / 'nonexistent_config.yaml'
        monkeypatch.setattr('src.config.CONFIG_PATH', nonexistent_path)

        with pytest.raises(FileNotFoundError):
            load_config()


# ── get_api_key() Tests ───────────────────────────────────────────────────────


class TestGetApiKey:
    """Tests for get_api_key() function."""

    def test_get_api_key_anthropic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify get_api_key('anthropic') returns ANTHROPIC_API_KEY env var value.

        Validates: Requirement 3.3
        """
        monkeypatch.setenv('ANTHROPIC_API_KEY', ANTHROPIC_KEY_VALUE)

        result = get_api_key('anthropic')

        assert result == ANTHROPIC_KEY_VALUE

    def test_get_api_key_openai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify get_api_key('openai') returns OPENAI_API_KEY env var value.

        Validates: Requirement 3.4
        """
        monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY_VALUE)

        result = get_api_key('openai')

        assert result == OPENAI_KEY_VALUE

    def test_get_api_key_huggingface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify get_api_key('huggingface') returns HF_TOKEN env var value.

        Validates: Requirement 3.5
        """
        monkeypatch.setenv('HF_TOKEN', HUGGINGFACE_TOKEN_VALUE)

        result = get_api_key('huggingface')

        assert result == HUGGINGFACE_TOKEN_VALUE

    def test_get_api_key_raises_value_error_for_unknown_service(self) -> None:
        """Verify get_api_key() raises ValueError for unknown service name.

        Validates: Requirement 3.6
        """
        with pytest.raises(ValueError):
            get_api_key(UNKNOWN_SERVICE_NAME)

    def test_get_api_key_returns_none_when_env_var_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify get_api_key() returns None when the env var is not set.

        Validates: Requirement 3.7
        """
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

        result = get_api_key('anthropic')

        assert result is None
