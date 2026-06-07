import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9$%./()'-]+")


def normalize_whitespace(text):
    return " ".join(text.split())


def tokenize_for_bm25(text):
    return TOKEN_PATTERN.findall(text.lower())


def snippet(text, max_chars=220):
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
