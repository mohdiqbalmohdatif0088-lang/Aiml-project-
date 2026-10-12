"""AI Writer with Text Summarization - Flask app."""
import os
import sqlite3
from datetime import datetime, timezone

from flask import Flask, g, jsonify, render_template, request

import summarizer
from readability import readability

DB_PATH = os.environ.get("HISTORY_DB", os.path.join(os.path.dirname(__file__), "history.db"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB request cap


# ---------- history (optional feature, stored in SQLite) ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute(
            """CREATE TABLE IF NOT EXISTS history (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 created_at TEXT NOT NULL,
                 model TEXT NOT NULL,
                 original TEXT NOT NULL,
                 summary TEXT NOT NULL,
                 stats TEXT NOT NULL)"""
        )
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


# ---------- routes ----------
@app.get("/")
def index():
    return render_template("index.html", models=list(summarizer.MODELS),
                           default_model=summarizer.DEFAULT_MODEL)


@app.post("/api/summarize")
def api_summarize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    model = data.get("model", summarizer.DEFAULT_MODEL)
    length = data.get("length", "medium")
    save = bool(data.get("save", True))

    try:
        result = summarizer.summarize(text, model=model, length=length)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # model download / runtime failures
        app.logger.exception("Summarization failed")
        return jsonify(error=f"The model could not run: {e}"), 500

    stats = {
        "original": readability(text),
        "summary": readability(result["summary"]),
    }
    stats["reduction_pct"] = round(
        100 * (1 - stats["summary"]["words"] / max(1, stats["original"]["words"])), 1)
    stats["chunks"] = result["chunks"]

    entry_id = None
    if save:
        import json
        cur = db().execute(
            "INSERT INTO history (created_at, model, original, summary, stats) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             result["model"], text, result["summary"], json.dumps(stats)))
        db().commit()
        entry_id = cur.lastrowid

    return jsonify(id=entry_id, summary=result["summary"], model=result["model"], stats=stats)


@app.get("/api/history")
def api_history():
    import json
    rows = db().execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT 50").fetchall()
    return jsonify([
        {"id": r["id"], "created_at": r["created_at"], "model": r["model"],
         "original": r["original"], "summary": r["summary"],
         "stats": json.loads(r["stats"])} for r in rows])


@app.delete("/api/history/<int:entry_id>")
def api_history_delete(entry_id):
    db().execute("DELETE FROM history WHERE id = ?", (entry_id,))
    db().commit()
    return "", 204


@app.delete("/api/history")
def api_history_clear():
    db().execute("DELETE FROM history")
    db().commit()
    return "", 204


if __name__ == "__main__":
    # Development server. For deployment use: gunicorn -w 1 -t 300 app:app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
