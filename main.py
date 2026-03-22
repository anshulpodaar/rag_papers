import os

from src.chunker import chunk_pages
from src.extractor import extract_text_by_page
from src.logger import get_logger
from src.section_detector import attach_sections

logger = get_logger(__name__)

if __name__ == '__main__':
	papers_dir_path = './papers'
	papers_list = sorted(os.listdir(papers_dir_path))

	for paper in papers_list:
		if not paper.endswith('.pdf'):
			continue

		logger.debug('\n\n\nProcessing: %s', paper)

		paper_path = os.path.join(papers_dir_path, paper)
		pages, full_text = extract_text_by_page(paper_path)

		if not pages:
			logger.warning('No text extracted from %s', paper)
			continue

		chunks = chunk_pages(pages)
		chunks = attach_sections(chunks)

		logger.info('Sample chunk: page=%d section=%s', chunks[0]['page'], chunks[0]['section'])
