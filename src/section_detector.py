import re

from src.logger import get_logger

logger = get_logger(__name__)

SECTION_HEADERS = [
	'abstract',
	'introduction',
	'background',
	'related work',
	'methods',
	'methodology',
	'experimental setup',
	'experiments',
	'results',
	'evaluation',
	'discussion',
	'conclusion',
	'future work',
	'references',
	'acknowledgements',
]

COMMON_SECTION_HEADERS = []


def detect_section(line: str) -> bool:
	"""Return True if a line looks like a section header."""
	line = line.strip()
	if not line or len(line) > 80:
		return False

	# Normalise spaced caps: 'I NTRODUCTION' -> 'INTRODUCTION'
	normalised = re.sub(r'(?<=[A-Z])\s+(?=[A-Z])', '', line)

	patterns = [
		r'^(abstract|references|acknowledgements?|appendix)$',  # known anchors
		r'^\d+(\.\d+)*\s+\w+',  # '1 Introduction', '3.2.1 Attention'
		r'^[A-Z]\.\d*\s+\w+',  # 'A.1 Trajectory Traces'
		r'^[A-Z]\s+\w+',  # 'A Further Results' (appendix letters)
	]

	for pattern in patterns:
		if re.match(pattern, normalised, re.IGNORECASE):
			return True

	return False


def normalise_header(line: str) -> str:
	"""Clean a detected header into a readable label."""
	# Remove spaced caps artifact
	line = re.sub(r'(?<=[A-Z])\s+(?=[A-Z])', '', line).strip()
	# Strip leading section numbers
	line = re.sub(r'^[\dA-Z]+(\.\d+)*\s+', '', line)
	return line.lower()


def attach_sections(chunks: list[dict]) -> list[dict]:
	"""
	Attach section labels to a list of chunks.

	Args:
		chunks: List of dicts {'page': int, 'text': str} returned by chunker.chunk_pages().

	Returns:
		The same list of chunks dicts with a 'section' (str) key added to each.
	"""


	current_section = 'unknown'

	for chunk in chunks:
		lines = chunk['text'].strip().split('\n')
		for line in lines:
			if detect_section(line):
				current_section = normalise_header(line)
				logger.debug('Detected section: %s', current_section)
				break
		chunk['section'] = current_section
		logger.debug('Section: %s | Length: %d', current_section, len(chunk['text']))

	logger.debug('Sections attached to %d chunks', len(chunks))
	return chunks
