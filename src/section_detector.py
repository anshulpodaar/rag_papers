import re

from src.logger import get_logger

logger = get_logger(__name__)

# ── Header detection ──────────────────────────────────────────────────────────

# Anchors: known fixed headers that appear in most papers
ANCHOR_HEADERS = {
    'abstract',
    'references',
    'acknowledgements',
    'acknowledgments',
    'appendix',
}

# Spaced caps artifact: 'I NTRODUCTION' → 'INTRODUCTION'
_SPACED_CAPS_RE = re.compile(r'(?<=[A-Z])\s+(?=[A-Z])')

# Section number patterns:
#   '1'  '1.2'  '1.2.3'  'A'  'A.1'  'A.1.2'  'B.2.1'
# Sub-sections must be .digits only (not .letters like .8Hr)
_SECTION_NUM_RE = re.compile(
    r'^(?P<num>[A-Z]|\d+)(?P<sub>(\.\d+)*)(?P<rest>\s+[A-Z]\S.*)$',
)


def _normalise_line(line: str) -> str:
    """Remove spaced-caps artifact and strip whitespace."""
    return _SPACED_CAPS_RE.sub('', line).strip()


# Require the title part to contain at least one real word (2+ letters)
_REAL_WORD_RE = re.compile(r'[a-zA-Z]{2,}')

# Axis label pattern: line is only numbers, spaces, and decimal points
_AXIS_LABEL_RE = re.compile(r'^[\d\s.\-×x]+$')

# Math/equation line: contains =, ^, {, }, \, ~
_MATH_RE = re.compile(r'[=\^{}\\\~]')

# Reject lines starting with punctuation or lowercase (not a header)
_STARTS_LOWERCASE_RE = re.compile(r'^[a-z,\.∈\(\)\[\]]')

# Reject lines ending with sentence punctuation (it's prose, not a header)
_SENTENCE_END_RE = re.compile(r'[a-z]\.$|[a-z]\,$|not\.$|could\.$')


def _is_header(line: str) -> bool:
    """
    Return True if a line looks like a section header.

    Args:
        line: A single line of text from the document.

    Returns:
        True if the line matches a known header pattern.
    """
    line = line.strip()
    if not line or len(line) > 80:
        return False

    # Reject axis labels (pure numbers and spaces)
    if _AXIS_LABEL_RE.match(line):
        logger.debug('Rejected axis label: %s', line)
        return False

    # Reject math/equation lines
    if _MATH_RE.search(line):
        logger.debug('Rejected math line: %s', line)
        return False

    # Reject lines starting with lowercase, comma, punctuation, or math symbols
    if _STARTS_LOWERCASE_RE.match(line):
        return False

    # Reject lines that end like a sentence (prose fragment, not a header)
    if _SENTENCE_END_RE.search(line.lower()):
        return False

    normalised = _normalise_line(line).lower()

    # Known anchor headers
    if normalised in ANCHOR_HEADERS:
        logger.debug('Matched anchor: %s', line)
        return True

    match = _SECTION_NUM_RE.match(_normalise_line(line).strip())
    if match:
        rest = match.group('rest').strip()
        words = rest.split()
        # Title part must contain at least one real word
        if (
            _REAL_WORD_RE.search(rest)
            and len(words) <= 6  # not a table row
            and len(rest) >= 4  # not a single abbreviation like 'Hr'
            and rest[0].isupper()  # title starts with capital
        ):
            logger.debug('Matched section number: %s', line)
            return True

    logger.debug('No match: %s', line)
    return False


def _parse_header(line: str) -> dict:
    """
    Extract section number and title from a header line.

    Args:
        line: A line already confirmed to be a header.

    Returns:
        Dict with keys 'number' (str | None) and 'title' (str).
    """
    normalised = _normalise_line(line)
    lower = normalised.lower()

    # Anchor headers have no number
    if lower in ANCHOR_HEADERS:
        return {
            'number': None,
            'title': lower
        }

    match = _SECTION_NUM_RE.match(normalised)
    if match:
        num = match.group('num') + match.group('sub')
        title = match.group('rest').strip().lower()
        return {
            'number': num,
            'title': title
        }

    return {
        'number': None,
        'title': lower
    }


def _is_subsection(number: str | None) -> bool:
    """
    Return True if a section number indicates a subsection (e.g. '3.1', 'A.2').

    Args:
        number: Section number string, or None for anchor headers.

    Returns:
        True if the number contains a dot (subsection level).
    """
    if number is None:
        return False
    return '.' in number


# ── Main pipeline function ────────────────────────────────────────────────────

def split_into_sections(lines: list[dict]) -> list[dict]:
    """
    Split a flat list of lines into hierarchical section blocks.

    Each section block groups lines under a top-level section and
    optional subsection. Subsections inherit their parent section label.

    Args:
        lines: List of dicts with keys 'line_idx' (int), 'text' (str),
            and 'page' (int), as returned by extractor.extract_lines().

    Returns:
        List of section block dicts, each with keys:
            'section'    (str): top-level section title
            'subsection' (str | None): subsection title, or None
            'lines'      (list[dict]): lines belonging to this block
    """
    blocks = []
    current_section = 'unknown'
    current_subsection = None
    current_lines = []

    for line_dict in lines:
        text = line_dict['text']

        if _is_header(text):
            # Save accumulated lines as a block before starting new one
            if current_lines:
                blocks.append(
                    {
                        'section': current_section,
                        'subsection': current_subsection,
                        'lines': current_lines,
                    }
                )
                current_lines = []

            parsed = _parse_header(text)
            number = parsed['number']
            title = parsed['title']

            if _is_subsection(number):
                # Subsection — inherit current section, update subsection
                current_subsection = title
            else:
                # Top-level section — reset both
                current_section = title
                current_subsection = None

            logger.debug(
                'Section: %s | Subsection: %s | Page: %d',
                current_section, current_subsection, line_dict['page']
            )
        else:
            current_lines.append(line_dict)

    # Flush remaining lines
    if current_lines:
        blocks.append(
            {
                'section': current_section,
                'subsection': current_subsection,
                'lines': current_lines,
            }
        )

    logger.debug(
        'Split into %d section blocks from %d lines',
        len(blocks), len(lines)
    )
    return blocks
