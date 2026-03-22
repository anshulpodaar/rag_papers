from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


def chunk_sections(blocks: list[dict]) -> list[dict]:
    """
    Split section blocks into overlapping text chunks.

    Each chunk inherits the section and subsection label from its block,
    and the page number of its first line.

    Args:
        blocks: List of section block dicts as returned by
            section_detector.split_into_sections().

    Returns:
        List of chunk dicts with keys:
            'text'       (str): chunk text
            'section'    (str): top-level section title
            'subsection' (str | None): subsection title, or None
            'page'       (int): page of the first line in this chunk
    """
    chunk_size = config['chunking']['chunk_size']
    chunk_overlap = config['chunking']['chunk_overlap']

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for block in blocks:
        if not block['lines']:
            continue

        block_text = '\n'.join(line['text'] for line in block['lines'])
        first_page = block['lines'][0]['page']

        for chunk_text in splitter.split_text(block_text):
            chunks.append({
                'text': chunk_text,
                'section': block['section'],
                'subsection': block['subsection'],
                'page': first_page,
            })

    logger.debug('Created %d chunks from %d section blocks', len(chunks), len(blocks))
    return chunks
