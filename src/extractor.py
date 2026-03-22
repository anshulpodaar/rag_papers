from pypdf import PdfReader

from src.logger import get_logger

logger = get_logger(__name__)


def extract_lines(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF as a flat, indexed list of lines.

    Each line carries its page number so page metadata is preserved
    through the pipeline even when we process by section rather than page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with keys:
            'line_idx' (int): global line index across the full document
            'text'     (str): the line text
            'page'     (int): 1-indexed page number
    """
    reader = PdfReader(pdf_path)
    lines = []
    line_idx = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ''
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                lines.append({
                    'line_idx': line_idx,
                    'text': stripped,
                    'page': page_num,
                })
                line_idx += 1

    logger.debug('Extracted %d lines from %s', len(lines), pdf_path)
    return lines
