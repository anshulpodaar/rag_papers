"""Unit tests for src/retriever.py — semantic search retrieval."""

from unittest.mock import MagicMock, call

import pytest

from tests.conftest import MOCK_EMBEDDING_VECTOR


# ── Constants ─────────────────────────────────────────────────────────────────

TEST_QUERY = 'What is the attention mechanism?'
DEFAULT_TOP_K = 5
TEST_SOURCE_FILTER = 'attention.pdf'
TEST_SECTION_FILTER = 'introduction'
TEST_SUBSECTION_FILTER = 'background'


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_retrieval_results(count: int = 3) -> list[dict]:
    """Create sample retrieval results."""
    return [
        {
            'text': f'Chunk {i} text content.',
            'source': 'attention.pdf',
            'section': 'introduction',
            'subsection': '',
            'page': i + 1,
            'score': round(0.9 - i * 0.1, 4),
        }
        for i in range(count)
    ]


def _make_source_chunks(count: int = 5) -> list[dict]:
    """Create sample source chunks for context window testing."""
    return [
        {
            'text': f'Source chunk {i} text.',
            'source': 'attention.pdf',
            'section': 'introduction',
            'subsection': '',
            'page': i + 1,
        }
        for i in range(count)
    ]


# ── Tests: retrieve() ────────────────────────────────────────────────────────


class TestRetrieve:
    """Tests for Retriever.retrieve() method."""

    def test_embeds_query_and_passes_to_vector_store(
        self, mock_embedder, mock_vector_store
    ):
        """Verify retrieve() embeds query and passes embedding to vector store query().

        Validates: Requirements 9.1
        """
        from src.retriever import Retriever

        retriever = Retriever(mock_embedder, mock_vector_store)
        retriever.retrieve(TEST_QUERY, top_k=DEFAULT_TOP_K)

        # The second _embed_query (which actually runs) uses embed()
        mock_embedder.embed.assert_called_once()
        embed_call_arg = mock_embedder.embed.call_args[0][0]
        assert embed_call_arg[0]['text'] == TEST_QUERY

        mock_vector_store.query.assert_called_once()
        query_kwargs = mock_vector_store.query.call_args[1]
        assert query_kwargs['embedding'] == MOCK_EMBEDDING_VECTOR
        assert query_kwargs['n_results'] == DEFAULT_TOP_K

    def test_filter_parameters_forwarded_to_vector_store(
        self, mock_embedder, mock_vector_store
    ):
        """Verify filter parameters are forwarded to vector store.

        Validates: Requirements 9.2
        """
        from src.retriever import Retriever

        retriever = Retriever(mock_embedder, mock_vector_store)
        retriever.retrieve(
            TEST_QUERY,
            top_k=3,
            source_filter=TEST_SOURCE_FILTER,
            section_filter=TEST_SECTION_FILTER,
            subsection_filter=TEST_SUBSECTION_FILTER,
        )

        query_kwargs = mock_vector_store.query.call_args[1]
        assert query_kwargs['source_filter'] == TEST_SOURCE_FILTER
        assert query_kwargs['section_filter'] == TEST_SECTION_FILTER
        assert query_kwargs['subsection_filter'] == TEST_SUBSECTION_FILTER

    def test_results_returned_as_is_from_vector_store(
        self, mock_embedder, mock_vector_store
    ):
        """Verify results are returned as-is from vector store.

        Validates: Requirements 9.3
        """
        from src.retriever import Retriever

        expected_results = _make_retrieval_results(2)
        mock_vector_store.query.return_value = expected_results

        retriever = Retriever(mock_embedder, mock_vector_store)
        results = retriever.retrieve(TEST_QUERY)

        assert results == expected_results

    def test_empty_retrieval_returns_empty_list(
        self, mock_embedder, mock_vector_store
    ):
        """Verify empty retrieval returns empty list.

        Validates: Requirements 9.6
        """
        from src.retriever import Retriever

        mock_vector_store.query.return_value = []

        retriever = Retriever(mock_embedder, mock_vector_store)
        results = retriever.retrieve(TEST_QUERY)

        assert results == []


# ── Tests: retrieve_with_context() ───────────────────────────────────────────


class TestRetrieveWithContext:
    """Tests for Retriever.retrieve_with_context() method."""

    def test_context_window_zero_matches_retrieve(
        self, mock_embedder, mock_vector_store
    ):
        """Verify retrieve_with_context() with context_window=0 matches retrieve().

        Validates: Requirements 9.4
        """
        from src.retriever import Retriever

        expected_results = _make_retrieval_results(2)
        mock_vector_store.query.return_value = expected_results

        retriever = Retriever(mock_embedder, mock_vector_store)
        results = retriever.retrieve_with_context(
            TEST_QUERY, top_k=2, context_window=0
        )

        assert results == expected_results
        # get_by_source should NOT be called when context_window=0
        mock_vector_store.get_by_source.assert_not_called()

    def test_context_window_positive_fetches_surrounding_chunks(
        self, mock_embedder, mock_vector_store
    ):
        """Verify retrieve_with_context() with context_window > 0 fetches surrounding chunks.

        Validates: Requirements 9.5
        """
        from src.retriever import Retriever

        # Set up: retrieve returns one result matching chunk index 2
        source_chunks = _make_source_chunks(5)
        retrieval_result = {
            'text': source_chunks[2]['text'],
            'source': 'attention.pdf',
            'section': 'introduction',
            'subsection': '',
            'page': 3,
            'score': 0.85,
        }
        mock_vector_store.query.return_value = [retrieval_result]
        mock_vector_store.get_by_source.return_value = source_chunks

        retriever = Retriever(mock_embedder, mock_vector_store)
        retriever.retrieve_with_context(
            TEST_QUERY, top_k=1, context_window=1
        )

        # Verify get_by_source was called to fetch surrounding chunks
        mock_vector_store.get_by_source.assert_called_with('attention.pdf')

    def test_empty_retrieval_returns_empty_list(
        self, mock_embedder, mock_vector_store
    ):
        """Verify retrieve_with_context() with empty results returns empty list.

        Validates: Requirements 9.6
        """
        from src.retriever import Retriever

        mock_vector_store.query.return_value = []

        retriever = Retriever(mock_embedder, mock_vector_store)
        results = retriever.retrieve_with_context(
            TEST_QUERY, top_k=5, context_window=1
        )

        assert results == []
        mock_vector_store.get_by_source.assert_not_called()
