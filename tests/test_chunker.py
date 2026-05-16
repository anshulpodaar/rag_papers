"""Unit tests for src/chunker.py — text chunking with metadata inheritance."""

import pytest

from src.chunker import chunk_sections


# ── Constants ─────────────────────────────────────────────────────────────────

EXPECTED_CHUNK_KEYS = {'text', 'section', 'subsection', 'page'}
TEST_CHUNK_SIZE = 200
TEST_CHUNK_OVERLAP = 20


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestChunkSections:
    """Tests for chunk_sections() function."""

    def test_output_contains_required_keys(self, sample_blocks):
        """Verify each output chunk contains text, section, subsection, page."""
        chunks = chunk_sections(sample_blocks)

        assert len(chunks) > 0
        for chunk in chunks:
            assert set(chunk.keys()) == EXPECTED_CHUNK_KEYS

    def test_metadata_inherited_from_source_block(self, sample_blocks):
        """Verify each chunk inherits section and subsection from its block."""
        chunks = chunk_sections(sample_blocks)

        # First block: section='abstract', subsection=None
        abstract_chunks = [c for c in chunks if c['section'] == 'abstract']
        assert len(abstract_chunks) > 0
        for chunk in abstract_chunks:
            assert chunk['subsection'] is None

        # Third block: section='introduction', subsection='background'
        bg_chunks = [
            c for c in chunks
            if c['section'] == 'introduction' and c['subsection'] == 'background'
        ]
        assert len(bg_chunks) > 0

    def test_page_equals_first_line_page_number(self, sample_blocks):
        """Verify page value equals the first line's page number in block."""
        chunks = chunk_sections(sample_blocks)

        # Abstract block first line is page 1
        abstract_chunks = [c for c in chunks if c['section'] == 'abstract']
        for chunk in abstract_chunks:
            assert chunk['page'] == 1

        # Background block first line is page 2
        bg_chunks = [
            c for c in chunks
            if c['subsection'] == 'background'
        ]
        for chunk in bg_chunks:
            assert chunk['page'] == 2

    def test_empty_lines_blocks_are_skipped(self):
        """Verify blocks with empty lines list produce no chunks."""
        blocks = [
            {
                'section': 'empty_section',
                'subsection': None,
                'lines': [],
            },
            {
                'section': 'real_section',
                'subsection': None,
                'lines': [
                    {'line_idx': 0, 'text': 'Some content here.', 'page': 1},
                ],
            },
        ]

        chunks = chunk_sections(blocks)

        sections = [c['section'] for c in chunks]
        assert 'empty_section' not in sections
        assert 'real_section' in sections

    def test_config_driven_chunk_size_and_overlap(self, sample_blocks):
        """Verify chunker reads chunk_size and chunk_overlap from config.

        The test config has chunk_size=200, chunk_overlap=20.
        With short text blocks, each block should produce exactly one chunk
        since the text is shorter than chunk_size.
        """
        chunks = chunk_sections(sample_blocks)

        # Each sample block has short text (< 200 chars), so one chunk per block
        assert len(chunks) == len(sample_blocks)

    def test_long_text_produces_multiple_overlapping_chunks(self):
        """Verify text longer than chunk_size produces multiple chunks."""
        # Create a block with text much longer than chunk_size (200)
        long_sentence = 'This is a sentence that contributes to the length. '
        long_text_lines = [
            {
                'line_idx': i,
                'text': long_sentence,
                'page': 1,
            }
            for i in range(20)  # 20 * ~52 chars = ~1040 chars >> 200
        ]

        blocks = [
            {
                'section': 'methods',
                'subsection': 'details',
                'lines': long_text_lines,
            },
        ]

        chunks = chunk_sections(blocks)

        # Should produce multiple chunks since total text >> chunk_size
        assert len(chunks) > 1

        # All chunks should inherit metadata
        for chunk in chunks:
            assert chunk['section'] == 'methods'
            assert chunk['subsection'] == 'details'
            assert chunk['page'] == 1

        # Verify overlap: consecutive chunks should share some text
        for i in range(len(chunks) - 1):
            current_end = chunks[i]['text'][-TEST_CHUNK_OVERLAP:]
            next_start = chunks[i + 1]['text'][:TEST_CHUNK_OVERLAP * 2]
            # The overlap means some text from end of current appears in next
            assert len(set(current_end.split()) & set(next_start.split())) > 0
