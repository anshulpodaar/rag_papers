from pypdf import PdfReader


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, one dict per page.

    Args:
    	pdf_path: path to PDF file (single PDF file only)

    Returns:
        List of {'page': int, 'text': str}
    """
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ''
        if text.strip():
            pages.append({'page': i, 'text': text})

    return pages
