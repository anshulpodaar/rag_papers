"""Unit tests for the main module (main.py)."""

from unittest.mock import MagicMock

import pytest

from main import ingest_papers, query_papers


# ── Constants ─────────────────────────────────────────────────────────────────

MOCK_LINES = [
    {'line_idx': 0, 'text': 'Abstract', 'page': 1},
    {'line_idx': 1, 'text': 'This paper presents a novel approach.', 'page': 1},
]

MOCK_BLOCKS = [
    {
        'section': 'abstract',
        'subsection': None,
        'lines': [
            {'line_idx': 1, 'text': 'This paper presents a novel approach.', 'page': 1},
        ],
    },
]

MOCK_CHUNKS = [
    {
        'text': 'This paper presents a novel approach.',
        'section': 'abstract',
        'subsection': None,
        'page': 1,
    },
]

MOCK_CHUNKS_EMBEDDED = [
    {
        'text': 'This paper presents a novel approach.',
        'section': 'abstract',
        'subsection': None,
        'page': 1,
        'embedding': [0.1] * 384,
    },
]

MOCK_ASK_RESULT = {
    'answer': 'The attention mechanism computes weighted sums.',
    'sources': [{'source': 'attention.pdf', 'sections': ['intro'], 'pages': [1]}],
    'model': 'claude-sonnet-4-20250514',
    'usage': {'input_tokens': 100, 'output_tokens': 50},
    'transparency': {},
}

TEST_PDF_FILES = ['paper_a.pdf', 'paper_b.pdf']


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_main_pipeline(monkeypatch):
    """Patch all main module pipeline dependencies and return mock objects."""
    mock_embedder_cls = MagicMock()
    mock_embedder_inst = MagicMock()
    mock_embedder_cls.return_value = mock_embedder_inst
    mock_embedder_inst.embed_chunks.return_value = MOCK_CHUNKS_EMBEDDED

    mock_store_cls = MagicMock()
    mock_store_inst = MagicMock()
    mock_store_cls.return_value = mock_store_inst
    mock_store_inst.count = 5

    mock_retriever_cls = MagicMock()
    mock_retriever_inst = MagicMock()
    mock_retriever_cls.return_value = mock_retriever_inst

    mock_qa_cls = MagicMock()
    mock_qa_inst = MagicMock()
    mock_qa_cls.return_value = mock_qa_inst
    mock_qa_inst.ask_with_sources.return_value = MOCK_ASK_RESULT

    monkeypatch.setattr('main.Embedder', mock_embedder_cls)
    monkeypatch.setattr('main.VectorStore', mock_store_cls)
    monkeypatch.setattr('main.Retriever', mock_retriever_cls)
    monkeypatch.setattr('main.QAEngine', mock_qa_cls)
    monkeypatch.setattr('main.extract_lines', lambda path: MOCK_LINES)
    monkeypatch.setattr('main.split_into_sections', lambda lines: MOCK_BLOCKS)
    monkeypatch.setattr('main.chunk_sections', lambda blocks: MOCK_CHUNKS)
    monkeypatch.setattr('os.listdir', lambda path: TEST_PDF_FILES)

    return {
        'embedder_cls': mock_embedder_cls,
        'embedder': mock_embedder_inst,
        'store_cls': mock_store_cls,
        'store': mock_store_inst,
        'retriever_cls': mock_retriever_cls,
        'retriever': mock_retriever_inst,
        'qa_cls': mock_qa_cls,
        'qa': mock_qa_inst,
    }


# ── ingest_papers Tests ───────────────────────────────────────────────────────


class TestIngestPapers:
    """Tests for ingest_papers() function."""

    def test_ingest_calls_pipeline_stages_in_sequence(
        self, monkeypatch, mock_main_pipeline
    ):
        """Verify ingest_papers calls extract, split, chunk, embed, upsert in order."""
        extract_calls = []
        split_calls = []
        chunk_calls = []

        def mock_extract(path):
            extract_calls.append(path)
            return MOCK_LINES

        def mock_split(lines):
            split_calls.append(lines)
            return MOCK_BLOCKS

        def mock_chunk(blocks):
            chunk_calls.append(blocks)
            return MOCK_CHUNKS

        monkeypatch.setattr('main.extract_lines', mock_extract)
        monkeypatch.setattr('main.split_into_sections', mock_split)
        monkeypatch.setattr('main.chunk_sections', mock_chunk)

        ingest_papers()

        # Both PDFs should be processed
        assert len(extract_calls) == 2
        assert len(split_calls) == 2
        assert len(chunk_calls) == 2

        embedder = mock_main_pipeline['embedder']
        store = mock_main_pipeline['store']
        assert embedder.embed_chunks.call_count == 2
        assert store.upsert.call_count == 2

    def test_ingest_skips_papers_with_no_extracted_text(
        self, monkeypatch, mock_main_pipeline
    ):
        """Verify ingest_papers skips papers that return no text."""
        call_count = {'extract': 0}

        def mock_extract_empty(path):
            call_count['extract'] += 1
            if 'paper_a.pdf' in path:
                return []  # No text extracted
            return MOCK_LINES

        monkeypatch.setattr('main.extract_lines', mock_extract_empty)

        ingest_papers()

        # extract_lines called for both, but only paper_b proceeds
        assert call_count['extract'] == 2

        embedder = mock_main_pipeline['embedder']
        store = mock_main_pipeline['store']
        # Only paper_b should be embedded and stored
        assert embedder.embed_chunks.call_count == 1
        assert store.upsert.call_count == 1


# ── query_papers Tests ────────────────────────────────────────────────────────


class TestQueryPapers:
    """Tests for query_papers() function."""

    def test_query_creates_components_and_calls_ask_with_sources(
        self, mock_main_pipeline
    ):
        """Verify query_papers creates pipeline and calls ask_with_sources."""
        question = 'What is the attention mechanism?'
        result = query_papers(question)

        # Verify components were created
        mock_main_pipeline['embedder_cls'].assert_called_once()
        mock_main_pipeline['store_cls'].assert_called_once()
        mock_main_pipeline['retriever_cls'].assert_called_once()
        mock_main_pipeline['qa_cls'].assert_called_once()

        # Verify ask_with_sources was called with the question
        qa = mock_main_pipeline['qa']
        qa.ask_with_sources.assert_called_once_with(question, top_k=5)

        # Verify result is returned
        assert result == MOCK_ASK_RESULT

    def test_query_passes_top_k_parameter(self, mock_main_pipeline):
        """Verify query_papers passes custom top_k to ask_with_sources."""
        question = 'How does self-attention work?'
        query_papers(question, top_k=10)

        qa = mock_main_pipeline['qa']
        qa.ask_with_sources.assert_called_once_with(question, top_k=10)
