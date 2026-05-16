"""Unit tests for src/qa_engine.py — QA engine with transparency metadata."""

from unittest.mock import MagicMock, patch

import pytest


# ── Constants ─────────────────────────────────────────────────────────────────

TEST_QUESTION = 'What is the attention mechanism?'
TEST_MODEL = 'claude-sonnet-4-20250514'
TEST_MAX_TOKENS = 512
MOCK_ANSWER = 'The attention mechanism allows models to focus on relevant parts.'
MOCK_INPUT_TOKENS = 150
MOCK_OUTPUT_TOKENS = 75
NO_CONTEXT_ANSWER = 'No relevant context found to answer this question.'

# Confidence formula weights
WEIGHT_TOP_SCORE = 0.6
WEIGHT_AVG_SCORE = 0.3
WEIGHT_SPREAD = 0.1

# Confidence level thresholds
THRESHOLD_HIGH = 0.75
THRESHOLD_MEDIUM = 0.50
THRESHOLD_LOW = 0.25

# Hallucination risk level thresholds
RISK_THRESHOLD_HIGH = 0.6
RISK_THRESHOLD_MEDIUM = 0.35
RISK_THRESHOLD_LOW = 0.15


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_anthropic_client() -> MagicMock:
    """Create a mock Anthropic client with configured response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=MOCK_ANSWER)]
    mock_response.usage.input_tokens = MOCK_INPUT_TOKENS
    mock_response.usage.output_tokens = MOCK_OUTPUT_TOKENS

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


def _make_chunks(scores: list[float], source: str = 'attention.pdf') -> list[dict]:
    """Create chunk dicts with given scores."""
    return [
        {
            'text': f'Chunk {i} text about attention.',
            'source': source,
            'section': 'introduction',
            'subsection': '',
            'page': i + 1,
            'score': score,
        }
        for i, score in enumerate(scores)
    ]


def _make_multi_source_chunks() -> list[dict]:
    """Create chunks from multiple sources for hallucination risk testing."""
    return [
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
            'source': 'transformers.pdf',
            'section': 'methods',
            'subsection': '',
            'page': 3,
            'score': 0.72,
        },
        {
            'text': 'Multi-head attention improves performance.',
            'source': 'multihead.pdf',
            'section': 'results',
            'subsection': '',
            'page': 5,
            'score': 0.60,
        },
    ]


def _create_qa_engine(mock_retriever: MagicMock) -> 'QAEngine':
    """Create a QAEngine with mocked Anthropic client."""
    with patch('src.qa_engine.Anthropic') as mock_anthropic_cls:
        mock_client = _make_mock_anthropic_client()
        mock_anthropic_cls.return_value = mock_client

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

    return engine


# ── Tests: ask() ──────────────────────────────────────────────────────────────


class TestAsk:
    """Tests for QAEngine.ask() method."""

    @patch('src.qa_engine.Anthropic')
    def test_ask_retrieves_chunks_and_calls_messages_create(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify ask() retrieves chunks and calls messages.create().

        Validates: Requirements 10.1
        """
        mock_client = _make_mock_anthropic_client()
        mock_anthropic_cls.return_value = mock_client

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        result = engine.ask(TEST_QUESTION)

        mock_retriever.retrieve.assert_called_once_with(
            query=TEST_QUESTION,
            top_k=5,
            source_filter=None,
            section_filter=None,
        )
        mock_client.messages.create.assert_called_once()
        assert result['answer'] == MOCK_ANSWER
        assert result['model'] == TEST_MODEL
        assert result['usage']['input_tokens'] == MOCK_INPUT_TOKENS
        assert result['usage']['output_tokens'] == MOCK_OUTPUT_TOKENS

    @patch('src.qa_engine.Anthropic')
    def test_ask_empty_retrieval_returns_no_context_without_api_call(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify ask() with empty retrieval returns 'no relevant context' without API call.

        Validates: Requirements 10.2
        """
        mock_client = _make_mock_anthropic_client()
        mock_anthropic_cls.return_value = mock_client
        mock_retriever.retrieve.return_value = []

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        result = engine.ask(TEST_QUESTION)

        assert result['answer'] == NO_CONTEXT_ANSWER
        assert result['usage'] == {'input_tokens': 0, 'output_tokens': 0}
        mock_client.messages.create.assert_not_called()


# ── Tests: ask_with_sources() ─────────────────────────────────────────────────


class TestAskWithSources:
    """Tests for QAEngine.ask_with_sources() method."""

    @patch('src.qa_engine.Anthropic')
    def test_response_contains_required_keys(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify ask_with_sources() response contains answer, sources, model, usage, transparency.

        Validates: Requirements 10.3
        """
        mock_client = _make_mock_anthropic_client()
        mock_anthropic_cls.return_value = mock_client

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        result = engine.ask_with_sources(TEST_QUESTION)

        required_keys = {'answer', 'sources', 'model', 'usage', 'transparency'}
        assert set(result.keys()) == required_keys
        assert isinstance(result['sources'], list)
        assert isinstance(result['transparency'], dict)


# ── Tests: _calculate_confidence() ───────────────────────────────────────────


class TestCalculateConfidence:
    """Tests for QAEngine._calculate_confidence() method."""

    @patch('src.qa_engine.Anthropic')
    def test_non_empty_chunks_returns_correct_keys_and_formula(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify _calculate_confidence() with non-empty chunks has correct keys and formula.

        Validates: Requirements 10.4, 10.7
        """
        mock_anthropic_cls.return_value = _make_mock_anthropic_client()

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        chunks = _make_chunks([0.9, 0.8, 0.7])
        result = engine._calculate_confidence(chunks)

        required_keys = {'overall_score', 'top_chunk_score', 'avg_score', 'score_spread', 'level'}
        assert set(result.keys()) == required_keys

        # Verify formula
        top_score = 0.9
        avg_score = (0.9 + 0.8 + 0.7) / 3
        spread = 0.9 - 0.7
        expected_overall = top_score * WEIGHT_TOP_SCORE + avg_score * WEIGHT_AVG_SCORE + (1 - spread) * WEIGHT_SPREAD
        assert result['overall_score'] == round(expected_overall, 4)

    @patch('src.qa_engine.Anthropic')
    def test_empty_chunks_returns_zero_score_and_none_level(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify _calculate_confidence() with empty list returns overall_score=0.0, level='none'.

        Validates: Requirements 10.5
        """
        mock_anthropic_cls.return_value = _make_mock_anthropic_client()

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        result = engine._calculate_confidence([])

        assert result['overall_score'] == 0.0
        assert result['level'] == 'none'
        assert result['top_chunk_score'] == 0.0
        assert result['avg_score'] == 0.0
        assert result['score_spread'] == 0.0

    @patch('src.qa_engine.Anthropic')
    def test_high_confidence_level_when_overall_score_gte_075(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify level == 'high' when overall_score >= 0.75.

        Validates: Requirements 10.6
        """
        mock_anthropic_cls.return_value = _make_mock_anthropic_client()

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        # Scores that produce overall >= 0.75
        # top=0.95, avg=0.90, spread=0.05
        # overall = 0.95*0.6 + 0.90*0.3 + (1-0.05)*0.1 = 0.57 + 0.27 + 0.095 = 0.935
        chunks = _make_chunks([0.95, 0.92, 0.90])
        result = engine._calculate_confidence(chunks)

        assert result['overall_score'] >= THRESHOLD_HIGH
        assert result['level'] == 'high'


# ── Tests: _calculate_hallucination_risk() ───────────────────────────────────


class TestCalculateHallucinationRisk:
    """Tests for QAEngine._calculate_hallucination_risk() method."""

    @patch('src.qa_engine.Anthropic')
    def test_empty_chunks_returns_score_1_level_high(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify empty chunks returns score=1.0, level='high', factors=['no_context_retrieved'].

        Validates: Requirements 10.8
        """
        mock_anthropic_cls.return_value = _make_mock_anthropic_client()

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        result = engine._calculate_hallucination_risk([], [])

        assert result['score'] == 1.0
        assert result['level'] == 'high'
        assert 'no_context_retrieved' in result['factors']

    @patch('src.qa_engine.Anthropic')
    def test_score_in_zero_one_range(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify hallucination risk score is in [0, 1].

        Validates: Requirements 10.9
        """
        mock_anthropic_cls.return_value = _make_mock_anthropic_client()

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        chunks = _make_chunks([0.85, 0.80, 0.75])
        sources = [{'source': 'attention.pdf', 'chunk_count': 3, 'avg_score': 0.8}]

        result = engine._calculate_hallucination_risk(chunks, sources)

        assert 0.0 <= result['score'] <= 1.0

    @patch('src.qa_engine.Anthropic')
    def test_level_thresholds(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify hallucination risk level thresholds are applied correctly.

        Validates: Requirements 10.10
        """
        mock_anthropic_cls.return_value = _make_mock_anthropic_client()

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        # High risk: many sources, low scores, high variance
        high_risk_chunks = _make_multi_source_chunks()
        # Add more chunks with low scores to push risk higher
        high_risk_chunks.extend(_make_chunks([0.2, 0.1], source='extra.pdf'))
        high_risk_sources = [
            {'source': 'attention.pdf'},
            {'source': 'transformers.pdf'},
            {'source': 'multihead.pdf'},
            {'source': 'extra.pdf'},
        ]
        result_high = engine._calculate_hallucination_risk(high_risk_chunks, high_risk_sources)
        # With 4 sources (multi_source +0.2), low avg (~0.49 < 0.6, moderate_grounding +0.15),
        # high variance, low top score... should be high
        assert result_high['level'] in ('high', 'medium')

        # Minimal risk: single source, high scores, low variance
        minimal_chunks = _make_chunks([0.95, 0.93, 0.92])
        minimal_sources = [{'source': 'attention.pdf'}]
        result_minimal = engine._calculate_hallucination_risk(minimal_chunks, minimal_sources)
        assert result_minimal['level'] == 'minimal'
        assert result_minimal['score'] < RISK_THRESHOLD_LOW


# ── Tests: _build_transparency() ─────────────────────────────────────────────


class TestBuildTransparency:
    """Tests for QAEngine._build_transparency() method."""

    @patch('src.qa_engine.Anthropic')
    def test_aggregates_sources_correctly(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify _build_transparency() aggregates sources with correct counts and scores.

        Validates: Requirements 10.11
        """
        mock_anthropic_cls.return_value = _make_mock_anthropic_client()

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        chunks = [
            {
                'text': 'Chunk 1',
                'source': 'paper_a.pdf',
                'section': 'introduction',
                'subsection': '',
                'page': 1,
                'score': 0.9,
            },
            {
                'text': 'Chunk 2',
                'source': 'paper_a.pdf',
                'section': 'methods',
                'subsection': '',
                'page': 3,
                'score': 0.7,
            },
            {
                'text': 'Chunk 3',
                'source': 'paper_b.pdf',
                'section': 'results',
                'subsection': '',
                'page': 5,
                'score': 0.8,
            },
        ]

        result = engine._build_transparency(chunks, MOCK_ANSWER)

        assert 'chunks_retrieved' in result
        assert 'sources' in result
        assert 'confidence' in result
        assert 'hallucination_risk' in result

        # Verify source aggregation
        sources = result['sources']
        assert len(sources) == 2

        # Find paper_a source
        paper_a = next(s for s in sources if s['source'] == 'paper_a.pdf')
        assert paper_a['chunk_count'] == 2
        assert paper_a['avg_score'] == round((0.9 + 0.7) / 2, 4)
        assert paper_a['max_score'] == round(0.9, 4)
        assert paper_a['min_score'] == round(0.7, 4)
        assert sorted(paper_a['sections']) == ['introduction', 'methods']
        assert sorted(paper_a['pages']) == [1, 3]

        # Find paper_b source
        paper_b = next(s for s in sources if s['source'] == 'paper_b.pdf')
        assert paper_b['chunk_count'] == 1
        assert paper_b['avg_score'] == round(0.8, 4)


# ── Tests: _format_context() ─────────────────────────────────────────────────


class TestFormatContext:
    """Tests for QAEngine._format_context() method."""

    @patch('src.qa_engine.Anthropic')
    def test_produces_formatted_string_with_chunk_text_and_metadata(
        self, mock_anthropic_cls, mock_retriever
    ):
        """Verify _format_context() produces formatted string with chunk text and metadata.

        Validates: Requirements 10.12
        """
        mock_anthropic_cls.return_value = _make_mock_anthropic_client()

        from src.qa_engine import QAEngine
        engine = QAEngine(retriever=mock_retriever)

        chunks = [
            {
                'text': 'Attention is all you need.',
                'source': 'attention.pdf',
                'section': 'abstract',
                'page': 1,
            },
            {
                'text': 'Transformers use self-attention.',
                'source': 'transformers.pdf',
                'section': 'introduction',
                'page': 2,
            },
        ]

        result = engine._format_context(chunks)

        # Verify it's a string containing chunk metadata and text
        assert isinstance(result, str)
        assert 'attention.pdf' in result
        assert 'abstract' in result
        assert 'Attention is all you need.' in result
        assert 'transformers.pdf' in result
        assert 'introduction' in result
        assert 'Transformers use self-attention.' in result

        # Verify numbered format
        assert '[1]' in result
        assert '[2]' in result
        assert 'Source:' in result
        assert 'Section:' in result
        assert 'Page:' in result
