import os

from src.extractor import extract_text_by_page

if __name__ == '__main__':
	papers_dir_path = './papers'
	papers_list = sorted(os.listdir(papers_dir_path))
	for paper in papers_list:
		if not paper.endswith('.pdf'):
			continue
		paper_path = os.path.join(papers_dir_path, paper)
		pages = extract_text_by_page(paper_path)
		print(f'\n{"*"*50}'
		      f'\nPaper: {paper}'
		      f'\nExtracted {len(pages)} pages.')
		if not pages:
			print(f'Warning: no text extracted from {paper}')
			continue
		print(f'\n\nSnippet:\n{pages[0]["text"][:500]}'
		      f'\n{"*"*50}\n')

