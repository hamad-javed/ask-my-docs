import html
import json
import sys
import webbrowser
from pathlib import Path

import chromadb
import numpy as np
from langchain_ollama import OllamaEmbeddings

import config

# Usage:
#   python3 visualize_db.py                       -> map of all chunks
#   python3 visualize_db.py "your question here"  -> also shows the question
#                                                    and the chunks it retrieves

question = " ".join(sys.argv[1:]).strip()

# 1. Read every chunk and its embedding from the saved DB
client = chromadb.PersistentClient(path=config.DB_DIRECTORY)
collection = client.get_collection(config.COLLECTION_NAME)
data = collection.get(include=["documents", "embeddings"])
texts = data["documents"]
vectors = np.array(data["embeddings"])

# 2. Embed the question and ask Chroma which chunks it would retrieve
retrieved = set()
if question:
    q_vector = OllamaEmbeddings(model=config.EMBEDDING_MODEL).embed_query(question)
    result = collection.query(query_embeddings=[q_vector], n_results=config.TOP_K)
    retrieved = set(result["ids"][0])
    vectors = np.vstack([vectors, q_vector])

# 3. Squash the 768-number embeddings down to 2D (PCA) so they can be plotted
centered = vectors - vectors.mean(axis=0)
_, _, components = np.linalg.svd(centered, full_matrices=False)
points = centered @ components[:2].T


def hover(text):
    # Wrap long chunks so the hover box stays readable
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) > 70:
            lines.append(line)
            line = ""
        line += word + " "
    lines.append(line)
    return "<br>".join(html.escape(l) for l in lines)


def label(text):
    first_line = text.strip().splitlines()[0]
    return first_line[:30] + ("…" if len(first_line) > 30 else "")


traces = []
for name, color, picked in [("Chunk", "#7c8db5", False), ("Retrieved chunk", "#e4572e", True)]:
    idx = [i for i, chunk_id in enumerate(data["ids"]) if (chunk_id in retrieved) == picked]
    if not idx:
        continue
    traces.append({
        "type": "scatter", "mode": "markers+text", "name": name,
        "x": points[idx, 0].tolist(), "y": points[idx, 1].tolist(),
        "text": [label(texts[i]) for i in idx], "textposition": "top center",
        "hovertext": [f"<b>Chunk {i + 1}</b><br>{hover(texts[i])}" for i in idx],
        "hoverinfo": "text", "marker": {"size": 16, "color": color},
    })
if question:
    traces.append({
        "type": "scatter", "mode": "markers+text", "name": "Your question",
        "x": [float(points[-1, 0])], "y": [float(points[-1, 1])],
        "text": ["❓ question"], "textposition": "bottom center",
        "hovertext": [hover(question)], "hoverinfo": "text",
        "marker": {"size": 22, "color": "#2a9d8f", "symbol": "star"},
    })

title = f'Question: "{html.escape(question)}"' if question else f"{len(texts)} chunks in Chroma"
layout = {
    "title": {"text": title}, "hovermode": "closest", "height": 700,
    "xaxis": {"title": "PCA 1", "zeroline": False}, "yaxis": {"title": "PCA 2", "zeroline": False},
    "legend": {"orientation": "h", "y": -0.12},
}

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Chroma DB map</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body {{ font-family: system-ui, sans-serif; margin: 24px; }}</style></head>
<body>
<p>Each dot is one chunk. Dots close together have similar meaning. Hover over a dot to read its text.</p>
<div id="chart"></div>
<script>Plotly.newPlot("chart", {json.dumps(traces)}, {json.dumps(layout)});</script>
</body></html>"""

out = Path("db_map.html").resolve()
out.write_text(page, encoding="utf-8")
print(f"Saved {out}")
webbrowser.open(out.as_uri())
