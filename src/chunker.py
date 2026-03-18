from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
		pages: list[dict],
		chunk_size: int = 600,
		chunk_overlap: int = 75
) -> list[dict]:
	"""
	Split extracted pages into overlapping text chunks.

	Args:
		pages: List of dicts with keys 'page' (int) and 'text' (str),
			as returned by extract_text_by_page().
		chunk_size: Maximum number of characters per chunk.
		chunk_overlap: Number of characters to overlap between chunks.

	Returns:
		List of dicts with keys 'page' (int), 'source' (str) and 'text' (str).
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

	return chunks
