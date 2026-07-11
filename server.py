#!/usr/bin/env python3
"""
server.py — serve ONLY the viewer/ folder on port 4700, plus POST /chat.

Standard library only. Usage:
    python3 build.py   # generate viewer/graph-data.js first
    python3 server.py  # then open http://localhost:4700

The Anthropic API key lives in config.json (project root) or the
ANTHROPIC_API_KEY environment variable. config.json is NOT inside viewer/,
so it can never be served to the browser.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 4700
ROOT = Path(__file__).parent
VIEWER_DIR = ROOT / "viewer"
CONFIG_PATH = ROOT / "config.json"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
TOP_K = 6
MAX_HISTORY_TURNS = 6          # user+assistant pairs kept per session
SESSION_TTL_SECONDS = 60 * 60  # drop sessions idle for an hour

STOPWORDS = frozenset(
    "a an the and or but if then than so of in on at to for from by with about into over "
    "is are was were be been being do does did has have had it its this that these those "
    "i you he she we they them his her their our your my me us what which who whom whose "
    "when where why how not no nor can could will would should may might must there here".split()
)

WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


# ---------------------------------------------------------------- notes index

def load_notes() -> list[dict]:
    """Parse viewer/graph-data.js back into note dicts (id, label, group, excerpt)."""
    raw = (VIEWER_DIR / "graph-data.js").read_text(encoding="utf-8")
    start, end = raw.index("{"), raw.rindex("}") + 1
    graph = json.loads(raw[start:end])
    return graph["nodes"]


def build_index(notes: list[dict]) -> list[dict]:
    """Precompute token bags; title tokens weigh extra at query time."""
    index = []
    for node in notes:
        index.append({
            "id": node["id"],
            "label": node["label"],
            "group": node["group"],
            "excerpt": node["excerpt"],
            "title_tokens": Counter(tokenize(node["label"])),
            "body_tokens": Counter(tokenize(node["excerpt"])),
        })
    return index


def score_notes(question: str, index: list[dict]) -> list[dict]:
    """Keyword overlap; a hit on the title counts 4x a hit in the body."""
    q_tokens = set(tokenize(question))
    scored = []
    for note in index:
        score = 0.0
        for tok in q_tokens:
            score += 4.0 * note["title_tokens"].get(tok, 0)
            score += 1.0 * min(note["body_tokens"].get(tok, 0), 3)  # cap repeats
        if score > 0:
            scored.append((score, note))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [note for _, note in scored[:TOP_K]]


# ---------------------------------------------------------------- config / llm

def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        cfg["api_key"] = env_key
    return cfg


def call_anthropic(cfg: dict, system: str, messages: list[dict]) -> str:
    payload = json.dumps({
        "model": cfg["model"],
        "max_tokens": 300,
        "system": system,
        "messages": messages,
    }).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": cfg["api_key"],
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return "".join(block.get("text", "") for block in data.get("content", [])).strip()


SYSTEM_TEMPLATE = """You are the voice of a personal knowledge galaxy. Answer the user's question using ONLY the notes provided below. Answer in 2-3 sentences. If the notes do not cover the question, say plainly that the notes don't cover it — do not guess or use outside knowledge.

NOTES:
{notes}"""


def format_notes(notes: list[dict]) -> str:
    return "\n\n".join(
        f"[{n['label']}] (topic: {n['group']})\n{n['excerpt']}" for n in notes
    )


# ---------------------------------------------------------------- sessions

SESSIONS: dict[str, dict] = {}  # sid -> {"messages": [...], "touched": ts}


def get_session(sid: str) -> dict:
    now = time.time()
    for key in [k for k, v in SESSIONS.items() if now - v["touched"] > SESSION_TTL_SECONDS]:
        del SESSIONS[key]
    session = SESSIONS.setdefault(sid, {"messages": [], "touched": now})
    session["touched"] = now
    return session


# ---------------------------------------------------------------- http handler

class ViewerHandler(SimpleHTTPRequestHandler):
    """Serves files from viewer/ only; POST /chat is the single API route."""

    notes_index: list[dict] = []

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[viewer] {self.address_string()} — {fmt % args}")

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/chat":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            question = str(req.get("question", "")).strip()[:500]
            sid = str(req.get("session", "default"))[:64]
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid request body."})
            return

        if not question:
            self._send_json(400, {"error": "Ask a question first."})
            return

        cfg = load_config()
        key = cfg.get("api_key", "")
        if not key or "PASTE_YOUR" in key:
            self._send_json(503, {
                "error": "No API key configured. Add it to config.json or set ANTHROPIC_API_KEY."
            })
            return

        top = score_notes(question, self.notes_index)
        if not top:
            self._send_json(200, {
                "answer": "The notes don't cover that — no note matched your question.",
                "nodes": [],
            })
            return

        session = get_session(sid)
        messages = session["messages"] + [{"role": "user", "content": question}]
        system = SYSTEM_TEMPLATE.format(notes=format_notes(top))

        try:
            answer = call_anthropic(cfg, system, messages)
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            print(f"[chat] Anthropic API error {err.code}: {detail}")
            self._send_json(502, {"error": f"Anthropic API error ({err.code}). Check the model name and key in config.json."})
            return
        except (urllib.error.URLError, TimeoutError) as err:
            print(f"[chat] network error: {err}")
            self._send_json(502, {"error": "Could not reach the Anthropic API."})
            return

        session["messages"] = (
            messages + [{"role": "assistant", "content": answer}]
        )[-MAX_HISTORY_TURNS * 2:]

        self._send_json(200, {"answer": answer, "nodes": [n["id"] for n in top]})


def main() -> None:
    if not VIEWER_DIR.is_dir():
        raise SystemExit(f"Viewer folder not found: {VIEWER_DIR}")
    if not (VIEWER_DIR / "graph-data.js").exists():
        raise SystemExit("viewer/graph-data.js missing — run `python3 build.py` first.")

    ViewerHandler.notes_index = build_index(load_notes())
    print(f"Indexed {len(ViewerHandler.notes_index)} notes for /chat.")

    handler = partial(ViewerHandler, directory=str(VIEWER_DIR))
    with ThreadingHTTPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"Knowledge Galaxy at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
