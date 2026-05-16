"""Unit tests for src/vector_store.py — ChromaDB vector store operations."""

from unittest.mock import MagicMock, patch, call

import pytest

from tests.conftest import EMBEDDING_DIMENSIONS, MOCK_EMBEDDING_VECTOR, TEST_CONFIG


# ── Constants ─────────────────────────────────────────────────────────────────

TEST_DB_PATH = './test_db'
TEST_COLLECTION_NAME = 'test_papers'
TEST_SOURCE = 'attention.pdf'
SCORE_PRECISION = 4


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_collection() -> MagicMock:
    """Create a mock ChromaDB collection with default return values."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.get.return_value = {
        'ids': [],
        'documents': [],
        'metadatas': [],
    }
    mock_collection.query.return_value = {
        'documents': [['chunk text']],
        'metadatas': [[{
            'source': TEST_SOURCE,
            'section': 'introduction',
            'subsection': '',
            'page': 1,
        }]],
        'distances': [[0.15]],
    }
    return mock_collection


def _make_mock_client(mock_collection: MagicMock) -> MagicMock:
    """Create a mock ChromaDB PersistentClient returning the given collection."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client


def _make_sample_chunks(count: int = 3) -> list[dict]:
    """Create sample chunks for upsert testing."""
    return [
        {
            'text': f'Chunk {i} text content.',
            'embedding': MOCK_EMBEDDING_VECTOR,
            'section': 'introduction',
            'subsection': None,
            'page': i + 1,
        }
        for i in range(count)
    ]


# ── Tests: Instantiation ─────────────────────────────────────────────────────


class TestVectorStoreInstantiation:
    """Tests for VectorStore constructor."""

    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.config', TEST_CONFIG)
    def test_creates_client_and_collection(self, mock_persistent_client):
        """Verify VectorStore creates PersistentClient and calls get_or_create_collection."""
        mock_collection = _make_mock_collection()
        mock_client = _make_mock_client(mock_collection)
        mock_persistent_client.return_value = mock_client

        from src.vector_store import VectorStore
        store = VectorStore()

        mock_persistent_client.assert_called_once_with(path=TEST_DB_PATH)
        mock_client.get_or_create_collection.assert_called_once_with(
            name=TEST_COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine'},
        )


# ── Tests: upsert() ──────────────────────────────────────────────────────────


class TestVectorStoreUpsert:
    """Tests for VectorStore.upsert() method."""

    @patch('src.vector_store.chromadb.PersistentClient')
    def test_upsert_deletes_existing_before_adding(self, mock_persistent_client):
        """Verify upsert() deletes existing chunks for source before adding new ones."""
        mock_collection = _make_mock_collection()
        # Simulate existing chunks for the source
        existing_ids = ['attention.pdf_chunk_0', 'attention.pdf_chunk_1']
        mock_collection.get.return_value = {
            'ids': existing_ids,
            'documents': ['old chunk 0', 'old chunk 1'],
            'metadatas': [
                {'source': TEST_SOURCE},
                {'source': TEST_SOURCE},
            ],
        }
        mock_client = _make_mock_client(mock_collection)
        mock_persistent_client.return_value = mock_client

        from src.vector_store import VectorStore
        store = VectorStore()

        chunks = _make_sample_chunks(2)
        result = store.upsert(chunks, TEST_SOURCE)

        # Verify delete was called with existing IDs
        mock_collection.delete.assert_called_once_with(ids=existing_ids)
        # Verify add was called with new chunks
        mock_collection.add.assert_called_once()
        assert result == 2


# ── Tests: query() ────────────────────────────────────────────────────────────


class TestVectorStoreQuery:
    """Tests for VectorStore.query() method."""

    @patch('src.vector_store.chromadb.PersistentClient')
    def test_query_delegates_to_collection(self, mock_persistent_client):
        """Verify query() delegates to mocked collection with correct parameters."""
        mock_collection = _make_mock_collection()
        mock_client = _make_mock_client(mock_collection)
        mock_persistent_client.return_value = mock_client

        from src.vector_store import VectorStore
        store = VectorStore()

        embedding = [0.5] * EMBEDDING_DIMENSIONS
        results = store.query(
            embedding=embedding,
            n_results=5,
            source_filter=TEST_SOURCE,
        )

        mock_collection.query.assert_called_once_with(
            query_embeddings=[embedding],
            n_results=5,
            where={'source': TEST_SOURCE},
            include=['documents', 'metadatas', 'distances'],
        )
        assert isinstance(results, list)


# ── Tests: _build_filter() ───────────────────────────────────────────────────


class TestBuildFilter:
    """Tests for VectorStore._build_filter() static method."""

    def test_all_none_returns_none(self):
        """Verify _build_filter() returns None when all parameters are None."""
        from src.vector_store import VectorStore

        result = VectorStore._build_filter(None, None, None)

        assert result is None

    def test_single_non_none_returns_simple_dict(self):
        """Verify _build_filter() returns simple dict when one parameter is non-None."""
        from src.vector_store import VectorStore

        result = VectorStore._build_filter('paper.pdf', None, None)

        assert result == {'source': 'paper.pdf'}

    def test_single_section_returns_simple_dict(self):
        """Verify _build_filter() returns simple dict for section-only filter."""
        from src.vector_store import VectorStore

        result = VectorStore._build_filter(None, 'introduction', None)

        assert result == {'section': 'introduction'}

    def test_multiple_non_none_returns_and_dict(self):
        """Verify _build_filter() returns $and dict when multiple parameters are non-None."""
        from src.vector_store import VectorStore

        result = VectorStore._build_filter('paper.pdf', 'introduction', None)

        assert '$and' in result
        assert {'source': 'paper.pdf'} in result['$and']
        assert {'section': 'introduction'} in result['$and']

    def test_all_non_none_returns_and_dict_with_three_conditions(self):
        """Verify _build_filter() returns $and dict with all three conditions."""
        from src.vector_store import VectorStore

        result = VectorStore._build_filter('paper.pdf', 'methods', 'background')

        assert '$and' in result
        assert len(result['$and']) == 3
        assert {'source': 'paper.pdf'} in result['$and']
        assert {'section': 'methods'} in result['$and']
        assert {'subsection': 'background'} in result['$and']


# ── Tests: _parse_results() ──────────────────────────────────────────────────


class TestParseResults:
    """Tests for VectorStore._parse_results() static method."""

    def test_score_conversion(self):
        """Verify score equals round(1 - distance, 4) for each result."""
        from src.vector_store import VectorStore

        distances = [0.15, 0.3, 0.0, 1.0, 0.7777]
        raw = {
            'documents': [['doc1', 'doc2', 'doc3', 'doc4', 'doc5']],
            'metadatas': [[
                {'source': 'a.pdf', 'section': 's1', 'subsection': '', 'page': 1},
                {'source': 'b.pdf', 'section': 's2', 'subsection': 'sub', 'page': 2},
                {'source': 'c.pdf', 'section': 's3', 'subsection': '', 'page': 3},
                {'source': 'd.pdf', 'section': 's4', 'subsection': '', 'page': 4},
                {'source': 'e.pdf', 'section': 's5', 'subsection': '', 'page': 5},
            ]],
            'distances': [distances],
        }

        results = VectorStore._parse_results(raw)

        assert len(results) == len(distances)
        for result, dist in zip(results, distances):
            expected_score = round(1 - dist, SCORE_PRECISION)
            assert result['score'] == expected_score

    def test_result_dict_contains_all_required_keys(self):
        """Verify each result dict contains text, source, section, subsection, page, score."""
        from src.vector_store import VectorStore

        raw = {
            'documents': [['Some text content']],
            'metadatas': [[{
                'source': 'test.pdf',
                'section': 'abstract',
                'subsection': 'overview',
                'page': 2,
            }]],
            'distances': [[0.25]],
        }

        results = VectorStore._parse_results(raw)

        assert len(results) == 1
        required_keys = {'text', 'source', 'section', 'subsection', 'page', 'score'}
        assert set(results[0].keys()) == required_keys
        assert results[0]['text'] == 'Some text content'
        assert results[0]['source'] == 'test.pdf'
        assert results[0]['section'] == 'abstract'
        assert results[0]['subsection'] == 'overview'
        assert results[0]['page'] == 2
        assert results[0]['score'] == round(1 - 0.25, SCORE_PRECISION)

    def test_empty_results(self):
        """Verify _parse_results() handles empty response correctly."""
        from src.vector_store import VectorStore

        raw = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]],
        }

        results = VectorStore._parse_results(raw)

        assert results == []


# ── Tests: get_by_source(), delete_source(), list_sources(), count ────────────


class TestVectorStoreOperations:
    """Tests for get_by_source(), delete_source(), list_sources(), and count."""

    @patch('src.vector_store.chromadb.PersistentClient')
    def test_get_by_source(self, mock_persistent_client):
        """Verify get_by_source() queries collection with source filter."""
        mock_collection = _make_mock_collection()
        mock_collection.get.return_value = {
            'ids': ['test.pdf_chunk_0'],
            'documents': ['Some document text'],
            'metadatas': [{
                'source': 'test.pdf',
                'section': 'abstract',
                'subsection': '',
                'page': 1,
            }],
        }
        mock_client = _make_mock_client(mock_collection)
        mock_persistent_client.return_value = mock_client

        from src.vector_store import VectorStore
        store = VectorStore()

        results = store.get_by_source('test.pdf')

        # The first call to get() is during __init__ (count log), then get_by_source
        get_calls = mock_collection.get.call_args_list
        # The get_by_source call should have where={'source': 'test.pdf'}
        assert any(
            c == call(where={'source': 'test.pdf'}, include=['documents', 'metadatas'])
            for c in get_calls
        )
        assert len(results) == 1
        assert results[0]['text'] == 'Some document text'
        assert results[0]['source'] == 'test.pdf'

    @patch('src.vector_store.chromadb.PersistentClient')
    def test_delete_source(self, mock_persistent_client):
        """Verify delete_source() removes all chunks for the given source."""
        mock_collection = _make_mock_collection()
        existing_ids = ['paper.pdf_chunk_0', 'paper.pdf_chunk_1']
        mock_collection.get.return_value = {
            'ids': existing_ids,
            'documents': ['doc1', 'doc2'],
            'metadatas': [{'source': 'paper.pdf'}, {'source': 'paper.pdf'}],
        }
        mock_client = _make_mock_client(mock_collection)
        mock_persistent_client.return_value = mock_client

        from src.vector_store import VectorStore
        store = VectorStore()

        count_deleted = store.delete_source('paper.pdf')

        mock_collection.delete.assert_called_with(ids=existing_ids)
        assert count_deleted == 2

    @patch('src.vector_store.chromadb.PersistentClient')
    def test_list_sources(self, mock_persistent_client):
        """Verify list_sources() returns sorted list of unique source filenames."""
        mock_collection = _make_mock_collection()
        mock_collection.get.return_value = {
            'ids': ['a_chunk_0', 'b_chunk_0', 'a_chunk_1'],
            'documents': ['d1', 'd2', 'd3'],
            'metadatas': [
                {'source': 'zebra.pdf'},
                {'source': 'alpha.pdf'},
                {'source': 'zebra.pdf'},
            ],
        }
        mock_client = _make_mock_client(mock_collection)
        mock_persistent_client.return_value = mock_client

        from src.vector_store import VectorStore
        store = VectorStore()

        sources = store.list_sources()

        assert sources == ['alpha.pdf', 'zebra.pdf']

    @patch('src.vector_store.chromadb.PersistentClient')
    def test_count_property(self, mock_persistent_client):
        """Verify count property returns the collection's count() value."""
        mock_collection = _make_mock_collection()
        mock_collection.count.return_value = 42
        mock_client = _make_mock_client(mock_collection)
        mock_persistent_client.return_value = mock_client

        from src.vector_store import VectorStore
        store = VectorStore()

        assert store.count == 42
        mock_collection.count.assert_called()
