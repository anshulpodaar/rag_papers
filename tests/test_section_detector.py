"""Unit tests for src/section_detector.py — header detection and section splitting."""

import pytest

from src.section_detector import (
    ANCHOR_HEADERS,
    _is_header,
    _is_subsection,
    _normalise_line,
    _parse_header,
    split_into_sections,
)


# ── _is_header() — Anchor headers ────────────────────────────────────────────


class TestIsHeaderAnchors:
    """Tests for _is_header() with known anchor headers."""

    @pytest.mark.parametrize('header', [
        'Abstract',
        'REFERENCES',
        'Acknowledgements',
        'ACKNOWLEDGMENTS',
        'Appendix',
    ])
    def test_anchor_headers_return_true(self, header):
        """Verify known anchor headers are detected regardless of case."""
        assert _is_header(header) is True


# ── _is_header() — Numbered headers ──────────────────────────────────────────


class TestIsHeaderNumbered:
    """Tests for _is_header() with numbered section headers."""

    @pytest.mark.parametrize('header', [
        '1 Introduction',
        '2 Related Work',
        '3.2 Methods',
        '4.1.1 Data Collection',
        'A.1 Supplementary Details',
    ])
    def test_numbered_headers_return_true(self, header):
        """Verify numbered section headers are detected."""
        assert _is_header(header) is True


# ── _is_header() — Rejection cases ───────────────────────────────────────────


class TestIsHeaderRejections:
    """Tests for _is_header() returning False on non-header lines."""

    @pytest.mark.parametrize('line', [
        '',
        '   ',
        '\t',
    ])
    def test_empty_and_whitespace_return_false(self, line):
        """Verify empty and whitespace-only strings return False."""
        assert _is_header(line) is False

    def test_line_over_80_chars_returns_false(self):
        """Verify lines longer than 80 characters return False."""
        long_line = '1 ' + 'A' * 80
        assert _is_header(long_line) is False

    @pytest.mark.parametrize('line', [
        '0.1 0.2 0.3 0.4',
        '1 2 3 4 5',
        '0.95',
    ])
    def test_axis_labels_return_false(self, line):
        """Verify axis labels (pure numbers/spaces) return False."""
        assert _is_header(line) is False

    @pytest.mark.parametrize('line', [
        'x = y + z',
        'f(x) = {a^2}',
        'E = mc^2',
    ])
    def test_math_lines_return_false(self, line):
        """Verify math/equation lines return False."""
        assert _is_header(line) is False

    @pytest.mark.parametrize('line', [
        'this starts with lowercase',
        'machine learning is great.',
    ])
    def test_lowercase_starting_lines_return_false(self, line):
        """Verify lines starting with lowercase return False."""
        assert _is_header(line) is False

    @pytest.mark.parametrize('line', [
        'This is a complete sentence.',
        'We show that the model converges.',
    ])
    def test_sentence_ending_lines_return_false(self, line):
        """Verify lines ending with sentence punctuation return False."""
        assert _is_header(line) is False


# ── _parse_header() ──────────────────────────────────────────────────────────


class TestParseHeader:
    """Tests for _parse_header() extracting number and title."""

    @pytest.mark.parametrize('header,expected_title', [
        ('Abstract', 'abstract'),
        ('REFERENCES', 'references'),
        ('Acknowledgements', 'acknowledgements'),
    ])
    def test_anchor_headers_have_no_number(self, header, expected_title):
        """Verify anchor headers parse with number=None and lowercased title."""
        result = _parse_header(header)
        assert result['number'] is None
        assert result['title'] == expected_title

    @pytest.mark.parametrize('header,expected_number,expected_title', [
        ('1 Introduction', '1', 'introduction'),
        ('3.2 Methods', '3.2', 'methods'),
        ('4.1.1 Data Collection', '4.1.1', 'data collection'),
        ('A.1 Supplementary Details', 'A.1', 'supplementary details'),
    ])
    def test_numbered_headers_parse_correctly(self, header, expected_number, expected_title):
        """Verify numbered headers parse with correct number and title."""
        result = _parse_header(header)
        assert result['number'] == expected_number
        assert result['title'] == expected_title


# ── _is_subsection() ─────────────────────────────────────────────────────────


class TestIsSubsection:
    """Tests for _is_subsection() detecting dotted numbers."""

    @pytest.mark.parametrize('number', [
        '3.1',
        '1.2.3',
        'A.1',
    ])
    def test_dotted_numbers_are_subsections(self, number):
        """Verify numbers with dots are identified as subsections."""
        assert _is_subsection(number) is True

    @pytest.mark.parametrize('number', [
        '1',
        '3',
        'A',
    ])
    def test_top_level_numbers_are_not_subsections(self, number):
        """Verify top-level numbers without dots are not subsections."""
        assert _is_subsection(number) is False

    def test_none_is_not_subsection(self):
        """Verify None returns False."""
        assert _is_subsection(None) is False


# ── _normalise_line() ─────────────────────────────────────────────────────────


class TestNormaliseLine:
    """Tests for _normalise_line() collapsing spaced-caps text."""

    @pytest.mark.parametrize('input_text,expected', [
        ('I NTRODUCTION', 'INTRODUCTION'),
        ('R EFERENCES', 'REFERENCES'),
        ('M ETHODS', 'METHODS'),
        ('A B S T R A C T', 'ABSTRACT'),
    ])
    def test_collapses_spaced_caps(self, input_text, expected):
        """Verify spaced-caps artifacts are collapsed."""
        assert _normalise_line(input_text) == expected

    def test_normal_text_unchanged(self):
        """Verify normal text passes through without modification."""
        assert _normalise_line('Normal text here') == 'Normal text here'

    def test_strips_whitespace(self):
        """Verify leading/trailing whitespace is stripped."""
        assert _normalise_line('  padded  ') == 'padded'


# ── split_into_sections() ────────────────────────────────────────────────────


class TestSplitIntoSections:
    """Tests for split_into_sections() hierarchical splitting."""

    def test_content_before_first_header_is_unknown(self):
        """Verify content before any header gets section='unknown'."""
        lines = [
            {'line_idx': 0, 'text': 'Some preamble text.', 'page': 1},
            {'line_idx': 1, 'text': 'More preamble.', 'page': 1},
            {'line_idx': 2, 'text': 'Abstract', 'page': 1},
            {'line_idx': 3, 'text': 'Abstract content here.', 'page': 1},
        ]

        blocks = split_into_sections(lines)

        assert blocks[0]['section'] == 'unknown'
        assert blocks[0]['subsection'] is None
        assert len(blocks[0]['lines']) == 2

    def test_subsection_blocks_inherit_parent_section(self):
        """Verify subsection blocks inherit the parent section name."""
        lines = [
            {'line_idx': 0, 'text': '1 Introduction', 'page': 1},
            {'line_idx': 1, 'text': 'Intro content.', 'page': 1},
            {'line_idx': 2, 'text': '1.1 Background', 'page': 1},
            {'line_idx': 3, 'text': 'Background content.', 'page': 1},
            {'line_idx': 4, 'text': '1.2 Motivation', 'page': 2},
            {'line_idx': 5, 'text': 'Motivation content.', 'page': 2},
        ]

        blocks = split_into_sections(lines)

        # First block: top-level intro content
        assert blocks[0]['section'] == 'introduction'
        assert blocks[0]['subsection'] is None

        # Second block: subsection inherits parent
        assert blocks[1]['section'] == 'introduction'
        assert blocks[1]['subsection'] == 'background'

        # Third block: another subsection inherits same parent
        assert blocks[2]['section'] == 'introduction'
        assert blocks[2]['subsection'] == 'motivation'

    def test_no_block_has_empty_lines(self):
        """Verify no output block has an empty lines list."""
        lines = [
            {'line_idx': 0, 'text': 'Abstract', 'page': 1},
            {'line_idx': 1, 'text': 'Content under abstract.', 'page': 1},
            {'line_idx': 2, 'text': '1 Introduction', 'page': 1},
            {'line_idx': 3, 'text': 'Intro text.', 'page': 1},
            {'line_idx': 4, 'text': '2 Methods', 'page': 2},
            {'line_idx': 5, 'text': 'Methods text.', 'page': 2},
        ]

        blocks = split_into_sections(lines)

        for block in blocks:
            assert len(block['lines']) > 0, (
                f'Block for section={block["section"]} has empty lines'
            )

    def test_consecutive_headers_without_content_produce_no_empty_blocks(self):
        """Verify consecutive headers without content between them don't create empty blocks."""
        lines = [
            {'line_idx': 0, 'text': '1 Introduction', 'page': 1},
            {'line_idx': 1, 'text': '2 Methods', 'page': 1},
            {'line_idx': 2, 'text': 'Methods content.', 'page': 1},
        ]

        blocks = split_into_sections(lines)

        for block in blocks:
            assert len(block['lines']) > 0

    def test_all_blocks_have_required_keys(self):
        """Verify every block has section, subsection, and lines keys."""
        lines = [
            {'line_idx': 0, 'text': 'Some text.', 'page': 1},
            {'line_idx': 1, 'text': 'Abstract', 'page': 1},
            {'line_idx': 2, 'text': 'Abstract body.', 'page': 1},
        ]

        blocks = split_into_sections(lines)

        for block in blocks:
            assert 'section' in block
            assert 'subsection' in block
            assert 'lines' in block
