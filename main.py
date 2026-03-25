import os

from src.chunker import chunk_sections
from src.embedder import Embedder
from src.extractor import extract_lines
from src.logger import get_logger
from src.section_detector import split_into_sections
from src.vector_store import VectorStore

logger = get_logger(__name__)

if __name__ == '__main__':
	papers_dir_path = './papers'
	papers_list = sorted(os.listdir(papers_dir_path))

	for paper in papers_list:
		if not paper.endswith('.pdf'):
			continue

		paper_path = os.path.join(papers_dir_path, paper)
		lines = extract_lines(paper_path)

		if not lines:
			logger.warning('No text extracted from %s', paper)
			continue

		blocks = split_into_sections(lines)
		chunks = chunk_sections(blocks)

		logger.info(
				'Paper: %s | Lines: %d | Blocks: %d | Chunks: %d',
				paper, len(lines), len(blocks), len(chunks)
		)

		embedder = Embedder()
		chunks_embedded = embedder.embed(chunks)
		logger.info('First chunk embedding length: %d', len(chunks_embedded[0]['embedding']))

		store = VectorStore()
		store.upsert(chunks, source = paper)
		logger.info('Total chunks in store: %d', store.count)
