"""Unit tests for the CLI module (src/cli.py)."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from src.cli import cmd_ask, cmd_delete, cmd_ingest, cmd_list, cmd_search, main


# ── Constants ─────────────────────────────────────────────────────────────────

MOCK_ANSWER = 'The attention mechanism works by computing weighted sums.'
MOCK_TRANSPARENCY = {
    'chunks_retrieved': [
        {
            'text': 'Attention is all you need.',
            'source': 'attention.pdf',
            'section': 'introduction',
            'subsection': '',
            'page': 1,
            'score': 0.85,
        },
    ],
    'sources': [
        {
            'source': 'attention.pdf',
            'sections': ['introduction'],
            'pages': [1],
            'chunk_count': 1,
            'avg_score': 0.85,
            'max_score': 0.85,
            'min_score': 0.85,
        },
    ],
    'confidence': {
        'overall_score': 0.82,
        'top_chunk_score': 0.85,
        'avg_score': 0.85,
        'score_spread': 0.0,
        'level': 'high',
    },
    'hallucination_risk': {
        'score': 0.0,
        'level': 'minimal',
        'factors': ['well_grounded'],
    },
}

MOCK_ASK_RESULT = {
    'answer': MOCK_ANSWER,
    'sources': MOCK_TRANSPARENCY['sources'],
    'model': 'claude-sonnet-4-20250514',
    'usage': {'input_tokens': 100, 'output_tokens': 50},
    'transparency': MOCK_TRANSPARENCY,
}

MOCK_SEARCH_RESULTS = [
    {
        'text': 'Attention is all you need.',
        'source': 'attention.pdf',
        'section': 'introduction',
        'subsection': '',
        'page': 1,
        'score': 0.85,
    },
    {
        'text': 'Transformers use self-attention.',
        'source': 'attention.pdf',
        'section': 'methods',
        'subsection': '',
        'page': 3,
        'score': 0.72,
    },
]

TEST_PDF_FILES = ['paper_a.pdf', 'paper_b.pdf', 'notes.txt']

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


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_cli_pipeline(monkeypatch):
    """Patch all CLI pipeline dependencies and return mock objects."""
    mock_embedder_cls = MagicMock()
    mock_embedder_inst = MagicMock()
    mock_embedder_cls.return_value = mock_embedder_inst
    mock_embedder_inst.embed_chunks.return_value = MOCK_CHUNKS_EMBEDDED

    mock_store_cls = MagicMock()
    mock_store_inst = MagicMock()
    mock_store_cls.return_value = mock_store_inst
    mock_store_inst.count = 5
    mock_store_inst.list_sources.return_value = ['paper_a.pdf', 'paper_b.pdf']
    mock_store_inst.get_by_source.return_value = [{'text': 'chunk1'}, {'text': 'chunk2'}]
    mock_store_inst.delete_source.return_value = 3

    mock_retriever_cls = MagicMock()
    mock_retriever_inst = MagicMock()
    mock_retriever_cls.return_value = mock_retriever_inst
    mock_retriever_inst.retrieve.return_value = MOCK_SEARCH_RESULTS

    mock_qa_cls = MagicMock()
    mock_qa_inst = MagicMock()
    mock_qa_cls.return_value = mock_qa_inst
    mock_qa_inst.ask_with_sources.return_value = MOCK_ASK_RESULT

    monkeypatch.setattr('src.cli.Embedder', mock_embedder_cls)
    monkeypatch.setattr('src.cli.VectorStore', mock_store_cls)
    monkeypatch.setattr('src.cli.Retriever', mock_retriever_cls)
    monkeypatch.setattr('src.cli.QAEngine', mock_qa_cls)
    monkeypatch.setattr('src.cli.extract_lines', lambda path: MOCK_LINES)
    monkeypatch.setattr('src.cli.split_into_sections', lambda lines: MOCK_BLOCKS)
    monkeypatch.setattr('src.cli.chunk_sections', lambda blocks: MOCK_CHUNKS)

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


# ── cmd_ingest Tests ──────────────────────────────────────────────────────────


class TestCmdIngest:
    """Tests for cmd_ingest() command."""

    def test_ingest_processes_pdfs_and_calls_upsert(
        self, monkeypatch, mock_cli_pipeline, capsys
    ):
        """Verify cmd_ingest processes PDF files and calls upsert."""
        monkeypatch.setattr('os.listdir', lambda path: TEST_PDF_FILES)

        args = argparse.Namespace(papers_dir=None)
        cmd_ingest(args)

        store = mock_cli_pipeline['store']
        # Only .pdf files should be processed (2 out of 3)
        assert store.upsert.call_count == 2

        captured = capsys.readouterr()
        assert 'paper_a.pdf' in captured.out
        assert 'paper_b.pdf' in captured.out

    def test_ingest_skips_non_pdf_files(
        self, monkeypatch, mock_cli_pipeline, capsys
    ):
        """Verify cmd_ingest skips files that are not PDFs."""
        monkeypatch.setattr('os.listdir', lambda path: ['notes.txt', 'readme.md'])

        args = argparse.Namespace(papers_dir=None)
        cmd_ingest(args)

        store = mock_cli_pipeline['store']
        store.upsert.assert_not_called()

        captured = capsys.readouterr()
        assert 'Ingested 0 papers' in captured.out


# ── cmd_ask Tests ─────────────────────────────────────────────────────────────


class TestCmdAsk:
    """Tests for cmd_ask() command."""

    def test_ask_with_valid_question_prints_answer(
        self, mock_cli_pipeline, capsys
    ):
        """Verify cmd_ask prints ANSWER: to stdout with a valid question."""
        args = argparse.Namespace(
            question=['What', 'is', 'attention?'],
            top_k=5,
            context_window=0,
            show_sources=False,
            verbose=False,
        )
        cmd_ask(args)

        captured = capsys.readouterr()
        assert 'ANSWER:' in captured.out
        assert MOCK_ANSWER in captured.out

    def test_ask_with_empty_question_prints_error_and_exits(
        self, mock_cli_pipeline, capsys
    ):
        """Verify cmd_ask prints error and exits with empty question."""
        args = argparse.Namespace(
            question=[''],
            top_k=5,
            context_window=0,
            show_sources=False,
            verbose=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_ask(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'Error' in captured.out

    def test_ask_with_whitespace_question_prints_error_and_exits(
        self, mock_cli_pipeline, capsys
    ):
        """Verify cmd_ask prints error and exits with whitespace-only question."""
        args = argparse.Namespace(
            question=['   ', '  '],
            top_k=5,
            context_window=0,
            show_sources=False,
            verbose=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_ask(args)

        assert exc_info.value.code == 1


# ── cmd_search Tests ──────────────────────────────────────────────────────────


class TestCmdSearch:
    """Tests for cmd_search() command."""

    def test_search_prints_results_with_scores(
        self, mock_cli_pipeline, capsys
    ):
        """Verify cmd_search prints results with scores."""
        args = argparse.Namespace(
            query=['attention', 'mechanism'],
            top_k=5,
            source=None,
            section=None,
        )
        cmd_search(args)

        captured = capsys.readouterr()
        assert 'Score: 0.8500' in captured.out
        assert 'Score: 0.7200' in captured.out
        assert 'attention.pdf' in captured.out


# ── cmd_list Tests ────────────────────────────────────────────────────────────


class TestCmdList:
    """Tests for cmd_list() command."""

    def test_list_prints_indexed_sources(self, mock_cli_pipeline, capsys):
        """Verify cmd_list prints the list of indexed sources."""
        args = argparse.Namespace()
        cmd_list(args)

        captured = capsys.readouterr()
        assert 'paper_a.pdf' in captured.out
        assert 'paper_b.pdf' in captured.out
        assert 'Indexed papers' in captured.out

    def test_list_empty_store_prints_message(self, mock_cli_pipeline, capsys):
        """Verify cmd_list prints message when no papers indexed."""
        mock_cli_pipeline['store'].list_sources.return_value = []

        args = argparse.Namespace()
        cmd_list(args)

        captured = capsys.readouterr()
        assert 'No papers indexed' in captured.out


# ── cmd_delete Tests ──────────────────────────────────────────────────────────


class TestCmdDelete:
    """Tests for cmd_delete() command."""

    def test_delete_valid_source_calls_delete_source(
        self, mock_cli_pipeline, capsys
    ):
        """Verify cmd_delete calls delete_source for a valid source."""
        args = argparse.Namespace(source='paper_a.pdf')
        cmd_delete(args)

        store = mock_cli_pipeline['store']
        store.delete_source.assert_called_once_with('paper_a.pdf')

        captured = capsys.readouterr()
        assert 'Deleted' in captured.out

    def test_delete_unknown_source_prints_error_and_exits(
        self, mock_cli_pipeline, capsys
    ):
        """Verify cmd_delete prints error and exits for unknown source."""
        args = argparse.Namespace(source='nonexistent.pdf')

        with pytest.raises(SystemExit) as exc_info:
            cmd_delete(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'not found' in captured.out


# ── main() Tests ──────────────────────────────────────────────────────────────


class TestMain:
    """Tests for main() entry point."""

    def test_main_no_arguments_prints_help_and_exits(self, monkeypatch, capsys):
        """Verify main() prints help and exits with no arguments."""
        monkeypatch.setattr('sys.argv', ['rag-papers'])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
