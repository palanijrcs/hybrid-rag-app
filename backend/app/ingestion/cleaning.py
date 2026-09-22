"""Clean raw text extracted from documents."""

import re

# Control characters that sometimes appear in PDF text
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# A word broken across a line: "recommen-\ndations"
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
# Three or more blank lines in a row
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
# Repeated spaces or tabs (but not newlines)
_EXTRA_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Tidy up extracted text without changing its meaning."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _EXTRA_SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()