"""Run every samples/*.txt file through the model and write samples/sample_outputs.md.

Usage:  python generate_samples.py
"""
import glob
import os

import summarizer
from readability import readability

HERE = os.path.dirname(os.path.abspath(__file__))
out = ["# Sample outputs\n"]

for path in sorted(glob.glob(os.path.join(HERE, "samples", "*.txt"))):
    text = open(path, encoding="utf-8").read()
    out.append(f"## {os.path.basename(path)}\n")
    for model in summarizer.MODELS:
        for length in ("short", "medium"):
            res = summarizer.summarize(text, model=model, length=length)
            a, b = readability(text), readability(res["summary"])
            out.append(f"### {model.upper()} · {length}\n")
            out.append(res["summary"] + "\n")
            out.append(f"- Words: {a['words']} → {b['words']}\n"
                       f"- Readability (Flesch): {a['flesch_reading_ease']} → "
                       f"{b['flesch_reading_ease']} ({b['label']})\n")

with open(os.path.join(HERE, "samples", "sample_outputs.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Wrote samples/sample_outputs.md")
