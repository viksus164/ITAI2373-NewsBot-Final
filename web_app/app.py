"""Flask frontend for NewsBot 2.0.

Run locally:
    python app.py

Then open:
    http://127.0.0.1:5000
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from newsbot_engine import NewsBotWebEngine


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "newsbot-local-demo-key")

newsbot = None


def get_newsbot() -> NewsBotWebEngine:
    """Load the NewsBot engine once and reuse it across requests."""
    global newsbot
    if newsbot is None:
        newsbot = NewsBotWebEngine()
    return newsbot


@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Render the project explanation page."""
    return render_template("about.html")


@app.route("/analyze", methods=["POST"])
def analyze_text():
    """Analyze pasted article text and render the results page."""
    text = request.form.get("text", "").strip()
    result = get_newsbot().analyze_complete(text)
    if not result.get("success"):
        return render_template("index.html", error=result.get("error", "Analysis failed."), text=text)
    return render_template("results.html", result=result, original_text=text)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON API endpoint for programmatic access."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    result = get_newsbot().analyze_complete(text)
    status = 200 if result.get("success") else 400
    return jsonify(result), status


@app.route("/health")
def health():
    """Simple health check for local testing."""
    return jsonify({"status": "ok", "engine_loaded": newsbot is not None})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
