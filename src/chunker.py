from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import get_logger

logger = get_logger(__name__)


def chunk_pages(
		pages: list[dict],
		chunk_size: int = 600,
		chunk_overlap: int = 75
) -> list[Optional[dict]]:
	"""
	Split extracted pages into overlapping text chunks.

	Args:
		pages: List of dicts {'page': int, 'text': str} returned by extractor.extract_text_by_page().
		chunk_size: Maximum number of characters per chunk.
		chunk_overlap: Number of characters to overlap between chunks.

	Returns:
		List of dicts {'page': int, 'text': str}
	"""
	splitter = RecursiveCharacterTextSplitter(
			chunk_size = chunk_size,
			chunk_overlap = chunk_overlap,
	)

	chunks = []
	for page in pages:
		for chunk_text in splitter.split_text(page['text']):
			chunks.append(
					{
						'page': page['page'],
						'text': chunk_text,
					}
			)

	logger.debug('Created %d chunks from %d pages', len(chunks), len(pages))
	return chunks
