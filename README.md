# AI Writer with Text Summarization

Flask web app that summarizes long articles with a BART or T5 model from Hugging Face Transformers.

## Features
- BART (`facebook/bart-large-cnn`) and T5 (`t5-base`) models, selectable in the UI
- Long articles are split into sentence-aligned chunks, summarized, then merged and re-summarized
- Short / medium / long summary settings
- Word counts, reduction %, Flesch reading ease and grade level
- Saved history (SQLite) with open and delete
- `.txt` file upload

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                       # http://localhost:5000
```
The first request downloads the model (BART is about 1.6 GB). Set `SUMMARIZER_MODEL=t5` to default to T5.

## Generate sample outputs
```bash
python generate_samples.py     # writes samples/sample_outputs.md
```

## Deploy
```bash
gunicorn -w 1 -t 300 -b 0.0.0.0:$PORT app:app
```
Use one worker so the model is loaded once, and a long timeout because the first request loads it.
Works on Render, Railway, Hugging Face Spaces (Docker) or any VM. A free CPU instance is slow on BART; use T5 there.

## Structure
```
app.py            Flask routes + SQLite history
summarizer.py     model loading, chunking, multi-pass summarization
readability.py    word count + Flesch scores (no dependencies)
templates/        index.html (UI)
samples/          sample article + generated outputs
```
