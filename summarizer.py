"""Summarization engine: BART / T5 via Hugging Face, with long-article chunking."""
import os
import re
import threading

from readability import count_words

MODELS = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-base",
}
DEFAULT_MODEL = os.environ.get("SUMMARIZER_MODEL", "bart")

# Fraction of the source to keep, per length setting.
LENGTH_RATIOS = {"short": 0.15, "medium": 0.30, "long": 0.45}

# BART handles ~1024 tokens (~700 words). Stay well under it.
CHUNK_WORDS = 450
MIN_WORDS = 40
MAX_WORDS = 20000

_pipes = {}
_lock = threading.Lock()


def _get_pipe(model_key: str):
    """Load each model once and reuse it across requests."""
    with _lock:
        if model_key not in _pipes:
            from transformers import pipeline  # imported lazily: slow import
            _pipes[model_key] = pipeline("summarization", model=MODELS[model_key])
        return _pipes[model_key]


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def chunk_text(text: str, max_words: int = CHUNK_WORDS):
    """Group whole sentences into chunks of at most `max_words` words."""
    chunks, current, size = [], [], 0
    for sent in split_sentences(text):
        words = sent.split()
        # A single enormous "sentence" (no punctuation): split it by words.
        pieces = [" ".join(words[i:i + max_words])
                  for i in range(0, len(words), max_words)] or [sent]
        for piece in pieces:
            n = len(piece.split())
            if current and size + n > max_words:
                chunks.append(" ".join(current))
                current, size = [], 0
            current.append(piece)
            size += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def _run_chunk(chunk: str, model_key: str, ratio: float) -> str:
    n = count_words(chunk)
    max_len = max(30, min(220, int(n * ratio * 1.4)))   # tokens ~ 1.3x words
    min_len = max(10, min(max_len - 5, int(max_len * 0.5)))
    out = _get_pipe(model_key)(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )
    return out[0]["summary_text"].strip()


def summarize(text: str, model: str = DEFAULT_MODEL, length: str = "medium") -> dict:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(MODELS)}.")
    if length not in LENGTH_RATIOS:
        raise ValueError(f"Unknown length '{length}'. Choose from: {', '.join(LENGTH_RATIOS)}.")
    text = text.strip()
    n_words = count_words(text)
    if n_words < MIN_WORDS:
        raise ValueError(f"Add at least {MIN_WORDS} words to summarize (got {n_words}).")
    if n_words > MAX_WORDS:
        raise ValueError(f"Text is too long ({n_words} words). The limit is {MAX_WORDS}.")

    ratio = LENGTH_RATIOS[length]
    chunks = chunk_text(text)
    n_chunks = len(chunks)
    passes = 1
    summary = " ".join(_run_chunk(c, model, ratio) for c in chunks)

    # Long article: summarize the chunk summaries again until they fit in one chunk
    # (max 3 extra passes), so the final result reads as one coherent summary.
    while len(chunks) > 1 and count_words(summary) > CHUNK_WORDS and passes < 4:
        chunks = chunk_text(summary)
        summary = " ".join(_run_chunk(c, model, max(ratio, 0.5)) for c in chunks)
        passes += 1
    return {
        "summary": summary,
        "model": MODELS[model],
        "chunks": n_chunks,
    }
