"""Unit tests for src/extractor.py — PDF text extraction."""

from unittest.mock import MagicMock, patch

import pytest

from src.extractor import extract_lines


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_page(text: str | None) -> MagicMock:
    """Create a mock PDF page with the given extract_text() return value."""
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def _make_mock_reader(pages: list[MagicMock]) -> MagicMock:
    """Create a mock PdfReader with the given pages."""
    reader = MagicMock()
    reader.pages = pages
    return reader


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestExtractLines:
    """Tests for extract_lines() function."""

    @patch('src.extractor.PdfReader')
    def test_returns_dicts_with_correct_keys(self, mock_pdf_reader):
        """Verify output dicts contain line_idx, text, and page keys."""
        pages = [_make_mock_page('Hello world\nSecond line')]
        mock_pdf_reader.return_value = _make_mock_reader(pages)

        result = extract_lines('dummy.pdf')

        assert len(result) == 2
        for line_dict in result:
            assert set(line_dict.keys()) == {'line_idx', 'text', 'page'}

    @patch('src.extractor.PdfReader')
    def test_page_numbers_are_1_indexed(self, mock_pdf_reader):
        """Verify page numbers start at 1, not 0."""
        pages = [
            _make_mock_page('Page one content'),
            _make_mock_page('Page two content'),
        ]
        mock_pdf_reader.return_value = _make_mock_reader(pages)

        result = extract_lines('dummy.pdf')

        assert result[0]['page'] == 1
        assert result[1]['page'] == 2

    @patch('src.extractor.PdfReader')
    def test_line_idx_is_globally_sequential(self, mock_pdf_reader):
        """Verify line_idx increments across pages without resetting."""
        pages = [
            _make_mock_page('Line A\nLine B'),
            _make_mock_page('Line C'),
        ]
        mock_pdf_reader.return_value = _make_mock_reader(pages)

        result = extract_lines('dummy.pdf')

        indices = [line['line_idx'] for line in result]
        assert indices == [0, 1, 2]

    @patch('src.extractor.PdfReader')
    def test_none_extract_text_pages_are_skipped(self, mock_pdf_reader):
        """Verify pages returning None from extract_text() are skipped."""
        pages = [
            _make_mock_page('First page'),
            _make_mock_page(None),
            _make_mock_page('Third page'),
        ]
        mock_pdf_reader.return_value = _make_mock_reader(pages)

        result = extract_lines('dummy.pdf')

        texts = [line['text'] for line in result]
        assert texts == ['First page', 'Third page']
        # Page numbers should still be correct (1-indexed per position)
        assert result[0]['page'] == 1
        assert result[1]['page'] == 3

    @patch('src.extractor.PdfReader')
    def test_empty_and_whitespace_lines_excluded(self, mock_pdf_reader):
        """Verify empty and whitespace-only lines are excluded."""
        pages = [_make_mock_page('Real content\n   \n\n  \t  \nAnother line')]
        mock_pdf_reader.return_value = _make_mock_reader(pages)

        result = extract_lines('dummy.pdf')

        texts = [line['text'] for line in result]
        assert texts == ['Real content', 'Another line']

    @patch('src.extractor.PdfReader')
    def test_pdf_with_no_extractable_text_returns_empty(self, mock_pdf_reader):
        """Verify PDF with no extractable text returns an empty list."""
        pages = [
            _make_mock_page(None),
            _make_mock_page(''),
            _make_mock_page('   \n\n  '),
        ]
        mock_pdf_reader.return_value = _make_mock_reader(pages)

        result = extract_lines('dummy.pdf')

        assert result == []

    @patch('src.extractor.PdfReader')
    def test_text_is_stripped(self, mock_pdf_reader):
        """Verify leading/trailing whitespace is stripped from line text."""
        pages = [_make_mock_page('  padded text  \n  another  ')]
        mock_pdf_reader.return_value = _make_mock_reader(pages)

        result = extract_lines('dummy.pdf')

        assert result[0]['text'] == 'padded text'
        assert result[1]['text'] == 'another'
