"""Shared test fixtures and config patching for all test modules."""

from unittest.mock import MagicMock

import pytest


# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_DIMENSIONS = 384
MOCK_EMBEDDING_VECTOR = [0.1] * EMBEDDING_DIMENSIONS


# ── Test Config ───────────────────────────────────────────────────────────────

TEST_CONFIG: dict = {
    'papers': {
        'path': './test_papers',
        'filetypes': ['.pdf'],
    },
    'vector_store': {
        'db_path': './test_db',
        'collection_name': 'test_papers',
        'n_results': 3,
    },
    'chunking': {
        'chunk_size': 200,
        'chunk_overlap': 20,
    },
    'embedding': {
        'model_name': 'all-MiniLM-L6-v2',
    },
    'claude': {
        'model': 'claude-sonnet-4-20250514',
        'max_tokens': 512,
    },
    'tracking': {
        'backend': 'json',
        'json': {'experiments_dir': './test_experiments'},
        'mlflow': {
            'experiment_name': 'test_rag',
            'tracking_uri': None,
        },
    },
}


# ── Config Patching (autouse) ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    """
    Patch src.config.config for every test automatically.

    Ensures no test reads the real config.yaml or .env file.
    """
    monkeypatch.setattr('src.config.config', TEST_CONFIG)


# ── Sample Data Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_lines() -> list[dict]:
    """
    Return sample extractor output lines with page tracking.

    Simulates output from extract_lines() for a 2-page document.
    """
    return [
        {'line_idx': 0, 'text': 'Abstract', 'page': 1},
        {'line_idx': 1, 'text': 'This paper presents a novel approach.', 'page': 1},
        {'line_idx': 2, 'text': 'We demonstrate significant improvements.', 'page': 1},
        {'line_idx': 3, 'text': '1 Introduction', 'page': 1},
        {'line_idx': 4, 'text': 'Machine learning has transformed many fields.', 'page': 1},
        {'line_idx': 5, 'text': 'Recent advances in transformers enable new capabilities.', 'page': 2},
        {'line_idx': 6, 'text': '1.1 Background', 'page': 2},
        {'line_idx': 7, 'text': 'The attention mechanism was introduced in 2017.', 'page': 2},
    ]


@pytest.fixture
def sample_blocks() -> list[dict]:
    """Return sample section detector output blocks."""
    return [
        {
            'section': 'abstract',
            'subsection': None,
            'lines': [
                {'line_idx': 1, 'text': 'This paper presents a novel approach.', 'page': 1},
                {'line_idx': 2, 'text': 'We demonstrate significant improvements.', 'page': 1},
            ],
        },
        {
            'section': 'introduction',
            'subsection': None,
            'lines': [
                {'line_idx': 4, 'text': 'Machine learning has transformed many fields.', 'page': 1},
                {'line_idx': 5, 'text': 'Recent advances in transformers enable new capabilities.', 'page': 2},
            ],
        },
        {
            'section': 'introduction',
            'subsection': 'background',
            'lines': [
                {'line_idx': 7, 'text': 'The attention mechanism was introduced in 2017.', 'page': 2},
            ],
        },
    ]


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Return sample chunker output chunks."""
    return [
        {
            'text': 'This paper presents a novel approach. We demonstrate significant improvements.',
            'section': 'abstract',
            'subsection': None,
            'page': 1,
        },
        {
            'text': 'Machine learning has transformed many fields. Recent advances in transformers enable new capabilities.',
            'section': 'introduction',
            'subsection': None,
            'page': 1,
        },
        {
            'text': 'The attention mechanism was introduced in 2017.',
            'section': 'introduction',
            'subsection': 'background',
            'page': 2,
        },
    ]


@pytest.fixture
def sample_chunks_embedded(sample_chunks) -> list[dict]:
    """Return sample chunks with 384-dimensional embedding vectors."""
    return [
        {**chunk, 'embedding': MOCK_EMBEDDING_VECTOR}
        for chunk in sample_chunks
    ]


# ── Mock Object Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_embedder():
    """
    Return a mock Embedder with configured return values.

    Uses spec=Embedder to catch interface drift.
    """
    from src.embedder import Embedder

    mock = MagicMock(spec=Embedder)
    mock.dimensions = EMBEDDING_DIMENSIONS
    mock.embed_text.return_value = MOCK_EMBEDDING_VECTOR
    mock.embed_batch.return_value = [MOCK_EMBEDDING_VECTOR]

    def embed_side_effect(chunks):
        for chunk in chunks:
            chunk['embedding'] = MOCK_EMBEDDING_VECTOR
        return chunks

    mock.embed.side_effect = embed_side_effect
    return mock


@pytest.fixture
def mock_vector_store():
    """
    Return a mock VectorStore with configured return values.

    Uses spec=VectorStore to catch interface drift.
    """
    from src.vector_store import VectorStore

    mock = MagicMock(spec=VectorStore)
    mock.count = 0
    mock.query.return_value = [
        {
            'text': 'Attention is all you need.',
            'source': 'attention.pdf',
            'section': 'introduction',
            'subsection': '',
            'page': 1,
            'score': 0.85,
        },
    ]
    mock.get_by_source.return_value = []
    mock.list_sources.return_value = ['attention.pdf']
    mock.upsert.return_value = 3
    mock.delete_source.return_value = 3
    return mock


@pytest.fixture
def mock_retriever():
    """
    Return a mock Retriever with configured return values.

    Uses spec=Retriever to catch interface drift.
    """
    from src.retriever import Retriever

    mock = MagicMock(spec=Retriever)
    mock.retrieve.return_value = [
        {
            'text': 'Attention is all you need.',
            'source': 'attention.pdf',
            'section': 'introduction',
            'subsection': '',
            'page': 1,
            'score': 0.85,
        },
    ]
    mock.retrieve_with_context.return_value = [
        {
            'text': 'Attention is all you need.',
            'source': 'attention.pdf',
            'section': 'introduction',
            'subsection': '',
            'page': 1,
            'score': 0.85,
        },
    ]
    return mock
