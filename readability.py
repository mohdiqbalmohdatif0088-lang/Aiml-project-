"""Word count and readability scoring (Flesch Reading Ease + Flesch-Kincaid grade).

Pure Python so it has no extra dependencies.
"""
import re

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _syllables(word: str) -> int:
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", word)))


def _band(score: float) -> str:
    if score >= 90: return "Very easy"
    if score >= 80: return "Easy"
    if score >= 70: return "Fairly easy"
    if score >= 60: return "Plain English"
    if score >= 50: return "Fairly difficult"
    if score >= 30: return "Difficult"
    return "Very difficult"


def readability(text: str) -> dict:
    words = _WORD_RE.findall(text)
    n_words = len(words)
    n_sents = max(1, len(_SENT_RE.findall(text)))
    if n_words == 0:
        return {"words": 0, "sentences": 0, "flesch_reading_ease": 0.0,
                "grade_level": 0.0, "label": "n/a"}
    n_syll = sum(_syllables(w) for w in words)
    wps = n_words / n_sents
    spw = n_syll / n_words
    ease = 206.835 - 1.015 * wps - 84.6 * spw
    grade = 0.39 * wps + 11.8 * spw - 15.59
    ease = max(0.0, min(100.0, ease))
    return {
        "words": n_words,
        "sentences": n_sents,
        "flesch_reading_ease": round(ease, 1),
        "grade_level": round(max(0.0, grade), 1),
        "label": _band(ease),
    }
