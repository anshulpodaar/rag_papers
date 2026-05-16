"""Unit tests for src/embedder.py — embedding with sentence-transformers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import EMBEDDING_DIMENSIONS


# ── Constants ─────────────────────────────────────────────────────────────────

CONFIGURED_MODEL_NAME = 'all-MiniLM-L6-v2'
CUSTOM_MODEL_NAME = 'custom-model-v1'


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_model(dimensions: int = EMBEDDING_DIMENSIONS) -> MagicMock:
    """Create a mock SentenceTransformer model with configured return values."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dimensions

    # encode() returns numpy arrays (single text or batch)
    single_vector = np.array([0.1] * dimensions)
    batch_vectors = np.array([[0.1] * dimensions, [0.2] * dimensions])

    def encode_side_effect(text, **kwargs):
        if isinstance(text, str):
            return single_vector
        return np.array([[0.1] * dimensions] * len(text))

    mock_model.encode.side_effect = encode_side_effect
    return mock_model


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestEmbedder:
    """Tests for Embedder class."""

    @patch('src.embedder.SentenceTransformer')
    def test_instantiation_calls_sentence_transformer_with_config_model(
        self, mock_st_class
    ):
        """Verify Embedder calls SentenceTransformer with configured model name."""
        mock_st_class.return_value = _make_mock_model()

        from src.embedder import Embedder
        embedder = Embedder()

        mock_st_class.assert_called_once_with(CONFIGURED_MODEL_NAME)

    @patch('src.embedder.SentenceTransformer')
    def test_fallback_to_config_model_name_when_no_argument(self, mock_st_class):
        """Verify Embedder uses config model name when no argument provided."""
        mock_st_class.return_value = _make_mock_model()

        from src.embedder import Embedder
        embedder = Embedder()

        mock_st_class.assert_called_once_with(CONFIGURED_MODEL_NAME)

    @patch('src.embedder.SentenceTransformer')
    def test_instantiation_with_custom_model_name(self, mock_st_class):
        """Verify Embedder uses provided model_name over config default."""
        mock_st_class.return_value = _make_mock_model()

        from src.embedder import Embedder
        embedder = Embedder(model_name=CUSTOM_MODEL_NAME)

        mock_st_class.assert_called_once_with(CUSTOM_MODEL_NAME)

    @patch('src.embedder.SentenceTransformer')
    def test_embed_text_returns_list_of_floats(self, mock_st_class):
        """Verify embed_text() returns a list of floats via mocked encode()."""
        mock_st_class.return_value = _make_mock_model()

        from src.embedder import Embedder
        embedder = Embedder()
        result = embedder.embed_text('Hello world')

        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIMENSIONS
        assert all(isinstance(x, float) for x in result)

    @patch('src.embedder.SentenceTransformer')
    def test_embed_batch_returns_list_of_embedding_vectors(self, mock_st_class):
        """Verify embed_batch() returns a list of embedding vectors."""
        mock_st_class.return_value = _make_mock_model()

        from src.embedder import Embedder
        embedder = Embedder()
        texts = ['First text', 'Second text', 'Third text']
        result = embedder.embed_batch(texts)

        assert isinstance(result, list)
        assert len(result) == len(texts)
        for vector in result:
            assert isinstance(vector, list)
            assert len(vector) == EMBEDDING_DIMENSIONS

    @patch('src.embedder.SentenceTransformer')
    def test_embed_adds_embedding_key_to_each_chunk(self, mock_st_class):
        """Verify embed() adds 'embedding' key to each chunk dict."""
        mock_st_class.return_value = _make_mock_model()

        from src.embedder import Embedder
        embedder = Embedder()
        chunks = [
            {'text': 'First chunk', 'section': 'intro', 'page': 1},
            {'text': 'Second chunk', 'section': 'methods', 'page': 2},
        ]
        result = embedder.embed(chunks)

        assert len(result) == 2
        for chunk in result:
            assert 'embedding' in chunk
            assert isinstance(chunk['embedding'], list)
            assert len(chunk['embedding']) == EMBEDDING_DIMENSIONS
            # Original keys preserved
            assert 'text' in chunk
            assert 'section' in chunk
            assert 'page' in chunk

    @patch('src.embedder.SentenceTransformer')
    def test_dimensions_property_returns_model_dimension(self, mock_st_class):
        """Verify dimensions property returns get_sentence_embedding_dimension()."""
        mock_model = _make_mock_model(dimensions=EMBEDDING_DIMENSIONS)
        mock_st_class.return_value = mock_model

        from src.embedder import Embedder
        embedder = Embedder()

        assert embedder.dimensions == EMBEDDING_DIMENSIONS
        mock_model.get_sentence_embedding_dimension.assert_called_once()
