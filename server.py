#!/usr/bin/env python3
"""
server.py — serve ONLY the viewer/ folder on port 4700, plus POST /chat.

Standard library only. Usage:
    python3 build.py   # generate viewer/graph-data.js first
    python3 server.py  # then open http://localhost:4700

The Gemini API key lives in config.json (project root) or a GEMINI_API_KEY /
ANTHROPIC_API_KEY environment variable / root .env file. config.json and the
.env files are NOT inside viewer/, so they can never be served to the browser.
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

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
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

def read_env_file_key(name: str) -> str:
    """Look up a key in local .env files (root only — never inside viewer/)."""
    for env_file in (ROOT / ".env.development.local", ROOT / ".env.local", ROOT / ".env"):
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "api_key")


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    for name in KEY_NAMES:
        env_key = os.environ.get(name, "").strip() or read_env_file_key(name)
        if env_key:
            cfg["api_key"] = env_key
            break
    return cfg


def call_gemini(cfg: dict, system: str, messages: list[dict]) -> str:
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    payload = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 1024},
    }).encode("utf-8")
    req = urllib.request.Request(
        GEMINI_URL.format(model=cfg["model"]),
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-goog-api-key": cfg["api_key"],
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    parts = data["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


SYSTEM_TEMPLATE = """You are the butler of Arnav's knowledge galaxy — his "second brain". You are a dry, impeccably polite British butler with a razor wit. Address him as "sir" occasionally (not every sentence); "sir Arnav" or simply "Arnav" where it suits the moment. One genuinely funny line beats three bland ones. Never grovel, never gush.

Two kinds of requests arrive:

1. Questions about the notes. Answer using ONLY the notes provided below: exactly ONE witty sentence, then the facts, briefly. NEVER recite or paraphrase the whole note back — it is already on his screen. If the notes don't cover it, admit it plainly (with dignity): do not guess or use outside knowledge.

2. Small talk, greetings, or jokes. Reply in character, briefly, and begin your reply with the exact token [CHAT] — this keeps the galaxy's camera still. Use [CHAT] ONLY when the request is genuinely not about the notes.

NOTES:
{notes}"""

CHAT_MARKER = "[CHAT]"


def format_notes(notes: list[dict]) -> str:
    if not notes:
        return "(no notes matched this question)"
    return "\n\n".join(
        f"[{n['label']}] (topic: {n['group']})\n{n['excerpt']}" for n in notes
    )


# ---------------------------------------------------------------- remember

NOTES_DIR = ROOT / "notes"
CAPTURES_DIR = NOTES_DIR / "captures"
GRAPH_DATA = VIEWER_DIR / "graph-data.js"
REMEMBER_RE = re.compile(r"^\s*remember\s+that\s*[,:]?\s*", re.IGNORECASE)
TITLE_WORDS = 6

WITTY_FALLBACK = "Filed and catalogued, sir — the galaxy grows by one star."

WITTY_SYSTEM = (
    "You are the dry, impeccably polite British butler of Arnav's knowledge galaxy. "
    "He has just dictated a new note, which you have filed. Confirm it in EXACTLY ONE "
    "short witty sentence. Address him as 'sir' if it suits. No preamble, no quotes."
)


def title_from_text(text: str) -> str:
    """A sensible title from the first few words, safe as a filename."""
    words = re.findall(r"[A-Za-z0-9']+", text)[:TITLE_WORDS]
    title = " ".join(words).strip() or "Untitled Capture"
    # Title-case but keep short connectives lowered mid-title
    small = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "are"}
    parts = [w if (w.lower() in small and i > 0) else w.capitalize()
             for i, w in enumerate(title.split())]
    return " ".join(parts)[:80]


def load_graph() -> dict:
    raw = GRAPH_DATA.read_text(encoding="utf-8")
    start, end = raw.index("{"), raw.rindex("}") + 1
    return json.loads(raw[start:end])


def save_graph(graph: dict) -> None:
    GRAPH_DATA.write_text(
        "const GRAPH = " + json.dumps(graph, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )


def remember_note(text: str, index: list[dict]) -> dict:
    """Write the note to notes/captures/, append it to graph-data.js with a
    STABLE id (= current node count — never re-sort, ids are array positions),
    and return the new node plus its links and anchor."""
    title = title_from_text(text)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    path = CAPTURES_DIR / f"{title}.md"
    if path.exists():  # same opening words twice — keep both
        path = CAPTURES_DIR / f"{title} ({time.strftime('%Y-%m-%d %H%M%S')}).md"
    path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")

    graph = load_graph()
    new_id = len(graph["nodes"])
    node = {"id": new_id, "label": path.stem, "group": "captures", "excerpt": text[:700]}

    # Link where one mentions the other's title; anchor = best keyword match.
    text_l = text.lower()
    links = [
        {"source": n["id"], "target": new_id}
        for n in graph["nodes"]
        if n["label"].lower() in text_l or title.lower() in n["excerpt"].lower()
    ]
    ranked = score_notes(text, index)
    anchor = ranked[0]["id"] if ranked else (links[0]["source"] if links else None)
    if not links and anchor is not None:
        links = [{"source": anchor, "target": new_id}]  # never leave a star orphaned

    graph["nodes"].append(node)
    graph["links"].extend(links)
    save_graph(graph)
    return {"node": node, "links": links, "anchor": anchor}


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
        if self.path == "/remember":
            self._handle_remember()
            return
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
                "error": "No API key configured. Add it to config.json or set GEMINI_API_KEY."
            })
            return

        # Even with zero matches, let the butler respond in character
        # (small talk, or a dignified admission that the notes are silent).
        top = score_notes(question, self.notes_index)

        session = get_session(sid)
        messages = session["messages"] + [{"role": "user", "content": question}]
        system = SYSTEM_TEMPLATE.format(notes=format_notes(top))

        try:
            answer = call_gemini(cfg, system, messages)
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            print(f"[chat] Gemini API error {err.code}: {detail}")
            self._send_json(502, {"error": f"Gemini API error ({err.code}). Check the model name and key in config.json."})
            return
        except (urllib.error.URLError, TimeoutError) as err:
            print(f"[chat] network error: {err}")
            self._send_json(502, {"error": "Could not reach the Gemini API."})
            return
        except (KeyError, IndexError) as err:
            print(f"[chat] unexpected Gemini response shape: {err}")
            self._send_json(502, {"error": "Gemini returned an unexpected response."})
            return

        # [CHAT] marks small talk: keep the camera still (no source nodes).
        small_talk = answer.startswith(CHAT_MARKER)
        if small_talk:
            answer = answer[len(CHAT_MARKER):].strip()

        session["messages"] = (
            messages + [{"role": "assistant", "content": answer}]
        )[-MAX_HISTORY_TURNS * 2:]

        node_ids = [] if small_talk else [n["id"] for n in top]
        self._send_json(200, {"answer": answer, "nodes": node_ids})

    def _handle_remember(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = REMEMBER_RE.sub("", str(req.get("text", ""))).strip()[:2000]
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid request body."})
            return

        if not text:
            self._send_json(400, {"error": "There was nothing to remember, sir."})
            return

        try:
            result = remember_note(text, self.__class__.notes_index)
        except OSError as err:
            print(f"[remember] write failed: {err}")
            self._send_json(500, {"error": "Could not write the note to disk."})
            return

        # the new note is immediately queryable via /chat
        self.__class__.notes_index = build_index(load_notes())

        # one witty confirmation line — canned fallback if the model is unavailable
        line = WITTY_FALLBACK
        cfg = load_config()
        if cfg.get("api_key") and "PASTE_YOUR" not in cfg["api_key"]:
            try:
                line = call_gemini(
                    cfg, WITTY_SYSTEM,
                    [{"role": "user", "content": f'The new note reads: "{text[:300]}"'}],
                ) or WITTY_FALLBACK
            except Exception as err:  # noqa: BLE001 — wit is optional, filing is not
                print(f"[remember] witty line failed, using fallback: {err}")

        result["answer"] = line
        print(f"[remember] filed '{result['node']['label']}' as node {result['node']['id']}")
        self._send_json(200, result)


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
