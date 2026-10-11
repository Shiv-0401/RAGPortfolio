"""Flask web interface for the RAGPortfolio chatbot."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flask import Flask, render_template, request, jsonify

from chain import ask
from retriever import indexed_document_count

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field."}), 400

    try:
        result = ask(question)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"answer": result["answer"], "sources": result["sources"]})


@app.route("/health")
def health():
    try:
        count = indexed_document_count()
        status = "ok"
    except Exception as exc:
        count = 0
        status = f"error: {exc}"

    return jsonify({"status": status, "indexed_documents": count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)
