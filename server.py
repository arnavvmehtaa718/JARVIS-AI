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
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("PORT", 4700))  # Render/most hosts inject PORT
ROOT = Path(__file__).parent
VIEWER_DIR = ROOT / "viewer"
CONFIG_PATH = ROOT / "config.json"
SETTINGS_PATH = ROOT / "settings.json"      # city etc. — root only, never served
ACTIVITY_LOG = ROOT / "jarvis-log.jsonl"    # time machine's memory — never served

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


# On 429 (rate limit), retry once after a pause, then fall back to the full
# flash model — it has its own separate free-tier quota. (Lite is primary:
# it answers in under a second where flash takes ~9s.)
FALLBACK_MODELS = ["gemini-flash-latest"]


def _gemini_once(model: str, api_key: str, system: str, messages: list[dict],
                 max_tokens: int = 400) -> str:
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    payload = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        # thinkingBudget 0 turns off the model's internal reasoning pass —
        # the single biggest latency win for these short butler replies.
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        GEMINI_URL.format(model=model),
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    parts = data["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def call_gemini(cfg: dict, system: str, messages: list[dict], max_tokens: int = 400) -> str:
    models = [cfg["model"]] + [m for m in FALLBACK_MODELS if m != cfg["model"]]
    last_err: urllib.error.HTTPError | None = None
    for i, model in enumerate(models):
        try:
            return _gemini_once(model, cfg["api_key"], system, messages, max_tokens)
        except urllib.error.HTTPError as err:
            if err.code != 429:
                raise
            last_err = err
            if i == 0:  # primary model: pause briefly and retry once before falling back
                print(f"[gemini] 429 on {model}, retrying in 3s")
                time.sleep(3)
                try:
                    return _gemini_once(model, cfg["api_key"], system, messages, max_tokens)
                except urllib.error.HTTPError as err2:
                    if err2.code != 429:
                        raise
                    last_err = err2
            print(f"[gemini] 429 on {model}, falling back")
    raise last_err  # every model rate-limited


SYSTEM_TEMPLATE = """You are JARVIS, butler of Arnav's knowledge galaxy — his "second brain". You are a dry, impeccably polite British butler with a razor wit. Address him as "sir" occasionally (not every sentence); "sir Arnav" or simply "Arnav" where it suits the moment. One genuinely funny line beats three bland ones. Never grovel, never gush.

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


def remember_note(text: str, index: list[dict], title: str | None = None) -> dict:
    """Write the note to notes/captures/, append it to graph-data.js with a
    STABLE id (= current node count — never re-sort, ids are array positions),
    and return the new node plus its links and anchor."""
    title = title or title_from_text(text)
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


# ---------------------------------------------------------------- activity log

def log_activity(kind: str, text: str, node_id: int | None = None) -> None:
    entry = {"ts": time.time(), "type": kind, "text": text[:400]}
    if node_id is not None:
        entry["node"] = node_id
    with ACTIVITY_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_activity(day: date) -> list[dict]:
    if not ACTIVITY_LOG.exists():
        return []
    entries = []
    for line in ACTIVITY_LOG.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            if date.fromtimestamp(e["ts"]) == day:
                entries.append(e)
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            continue
    return entries


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_when(text: str) -> date:
    """'last tuesday', 'yesterday', '3 days ago', 'wednesday' → a date. Defaults to today."""
    t = text.lower()
    today = date.today()
    if "yesterday" in t:
        return today - timedelta(days=1)
    m = re.search(r"(\d+)\s+days?\s+ago", t)
    if m:
        return today - timedelta(days=int(m.group(1)))
    for i, day in enumerate(WEEKDAYS):
        if day in t:
            delta = (today.weekday() - i) % 7
            return today - timedelta(days=delta or 7)  # bare weekday = most recent past one
    return today


# ---------------------------------------------------------------- settings / weather

def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode(city: str) -> dict | None:
    data = fetch_json(
        "https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name="
        + urllib.parse.quote(city)
    )
    hits = data.get("results") or []
    if not hits:
        return None
    hit = hits[0]
    return {
        "city": hit["name"],
        "region": hit.get("admin1", ""),
        "country": hit.get("country", ""),
        "lat": hit["latitude"],
        "lon": hit["longitude"],
    }


WMO = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "torrential showers", 95: "a thunderstorm",
    96: "a thunderstorm with hail", 99: "a thunderstorm with hail",
}


def fetch_weather(lat: float, lon: float) -> dict:
    data = fetch_json(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        "&timezone=auto&forecast_days=1"
    )
    cur, daily = data["current"], data["daily"]
    return {
        "now_c": round(cur["temperature_2m"]),
        "feels_c": round(cur["apparent_temperature"]),
        "sky": WMO.get(cur["weather_code"], "indeterminate skies"),
        "wind_kmh": round(cur["wind_speed_10m"]),
        "high_c": round(daily["temperature_2m_max"][0]),
        "low_c": round(daily["temperature_2m_min"][0]),
        "rain_pct": daily["precipitation_probability_max"][0],
    }


BRIEFING_SYSTEM = (
    "You are JARVIS, Arnav's dry, impeccably polite British butler. Compose his spoken "
    "briefing in 2-4 sentences: greet him by the time of day, give the weather plainly "
    "(temperature, sky, high/low, rain chance if notable), then the state of his second "
    "brain (note count, plus recent activity if any). One witty line maximum. No lists, "
    "no markdown — it will be read aloud."
)

TIMEMACHINE_SYSTEM = (
    "You are JARVIS, Arnav's dry British butler, reporting what he was doing on a given "
    "day based on his activity log. Summarize in 2-3 spoken sentences — what he asked "
    "about, what he filed. ONE witty observation maximum. If the log is empty, say so "
    "with dignity. No lists, no markdown."
)

# ---------------------------------------------------------------- agent hands

AGENT_PLAN_SYSTEM = (
    "You are JARVIS, Arnav's dry British butler. He has given you a task (research or "
    "repurposing content from his notes). In 1-2 sentences, state precisely what you "
    "intend to produce — then end with exactly: Shall I proceed, sir? Just say \"do it\". "
    "Do NOT do the task yet."
)

AGENT_EXEC_SYSTEM = (
    "You are JARVIS, Arnav's dry British butler, now executing an approved task. Using "
    "ONLY the notes below where they are relevant, produce the deliverable: clear, "
    "useful, and concise (under 250 words). Plain text only — no markdown syntax.\n\n"
    "NOTES:\n{notes}"
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

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path == "/settings":
            s = load_settings()
            self._send_json(200, {"city": s.get("city", ""), "region": s.get("region", "")})
            return
        if self.path == "/briefing":
            self._handle_briefing()
            return
        super().do_GET()

    def do_POST(self):
        routes = {
            "/remember": self._handle_remember,
            "/settings": self._handle_settings,
            "/timemachine": self._handle_timemachine,
            "/agent": self._handle_agent,
        }
        if self.path in routes:
            routes[self.path]()
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
            if err.code == 429:
                self._send_json(429, {"error": "My apologies, sir — the free Gemini quota is spent for the moment. Give it a minute (or a day, if the daily limit is hit) and try again."})
            else:
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

        log_activity("chat", question)
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
        log_activity("remember", text, result["node"]["id"])
        print(f"[remember] filed '{result['node']['label']}' as node {result['node']['id']}")
        self._send_json(200, result)

    # ------------------------------------------------------------ settings

    def _handle_settings(self):
        try:
            city = str(self._read_body().get("city", "")).strip()[:80]
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid request body."})
            return
        if not city:
            self._send_json(400, {"error": "A city, sir — I can hardly brief you on the weather of nowhere."})
            return
        try:
            place = geocode(city)
        except (urllib.error.URLError, TimeoutError, KeyError) as err:
            print(f"[settings] geocoding failed: {err}")
            self._send_json(502, {"error": "The geocoding service is unreachable at present."})
            return
        if not place:
            self._send_json(404, {"error": f'I searched the atlas twice, sir — no "{city}" to be found. Try again?'})
            return
        save_settings(place)
        where = place["city"] + (f", {place['country']}" if place["country"] else "")
        self._send_json(200, {
            "answer": f"Noted, sir. {where} it is — I shall keep an eye on its skies.",
            "city": place["city"],
        })

    # ------------------------------------------------------------ briefing

    def _handle_briefing(self):
        settings = load_settings()
        if not settings.get("lat"):
            self._send_json(409, {"error": "no_city"})
            return
        try:
            wx = fetch_weather(settings["lat"], settings["lon"])
        except (urllib.error.URLError, TimeoutError, KeyError) as err:
            print(f"[briefing] weather fetch failed: {err}")
            self._send_json(502, {"error": "The meteorological service is sulking, sir. Try again shortly."})
            return

        hour = datetime.now().hour
        tod = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        today_acts = read_activity(date.today())
        facts = (
            f"Time of day: {tod}. City: {settings['city']}. "
            f"Weather now: {wx['now_c']}°C (feels {wx['feels_c']}°C), {wx['sky']}, wind {wx['wind_kmh']} km/h. "
            f"Today: high {wx['high_c']}°C, low {wx['low_c']}°C, rain chance {wx['rain_pct']}%. "
            f"Second brain: {len(self.notes_index)} notes indexed. "
            f"Activity today so far: {len(today_acts)} interactions."
        )

        answer = (
            f"Good {tod}, sir. {settings['city']} stands at {wx['now_c']}°C with {wx['sky']}; "
            f"expect a high of {wx['high_c']} and a low of {wx['low_c']}, rain chance {wx['rain_pct']} percent. "
            f"Your second brain holds {len(self.notes_index)} notes, all present and accounted for."
        )
        cfg = load_config()
        if cfg.get("api_key") and "PASTE_YOUR" not in cfg["api_key"]:
            try:
                answer = call_gemini(cfg, BRIEFING_SYSTEM, [{"role": "user", "content": facts}]) or answer
            except Exception as err:  # noqa: BLE001 — template fallback is fine
                print(f"[briefing] model unavailable, using template: {err}")
        self._send_json(200, {"answer": answer})

    # ------------------------------------------------------------ time machine

    def _handle_timemachine(self):
        try:
            query = str(self._read_body().get("query", "")).strip()[:300]
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid request body."})
            return

        day = parse_when(query)
        entries = read_activity(day)
        day_label = day.strftime("%A, %B %d")
        nodes = [e["node"] for e in entries if e.get("type") == "remember" and "node" in e]

        if not entries:
            self._send_json(200, {
                "answer": f"The log for {day_label} is a blank page, sir — either a day of rest or a day of secrets.",
                "nodes": [],
            })
            return

        lines = [
            f"{datetime.fromtimestamp(e['ts']).strftime('%H:%M')} — "
            f"{'filed a note' if e['type'] == 'remember' else 'asked' if e['type'] == 'chat' else e['type']}: {e['text']}"
            for e in entries
        ]
        answer = f"On {day_label}: " + "; ".join(lines[:8])
        cfg = load_config()
        if cfg.get("api_key") and "PASTE_YOUR" not in cfg["api_key"]:
            try:
                answer = call_gemini(
                    cfg, TIMEMACHINE_SYSTEM,
                    [{"role": "user", "content": f"Day: {day_label}\nLog:\n" + "\n".join(lines[:30])}],
                ) or answer
            except Exception as err:  # noqa: BLE001
                print(f"[timemachine] model unavailable, using raw log: {err}")
        self._send_json(200, {"answer": answer, "nodes": nodes[:6]})

    # ------------------------------------------------------------ agent hands

    def _handle_agent(self):
        try:
            req = self._read_body()
            task = str(req.get("task", "")).strip()[:500]
            sid = str(req.get("session", "default"))[:64]
            confirm = bool(req.get("confirm"))
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid request body."})
            return

        session = get_session(sid)
        cfg = load_config()
        if not cfg.get("api_key") or "PASTE_YOUR" in cfg["api_key"]:
            self._send_json(503, {"error": "No API key configured."})
            return

        if not confirm:
            # stage 1: propose a plan, await "do it"
            if not task:
                self._send_json(400, {"error": "Task required."})
                return
            top = score_notes(task, self.notes_index)
            context = f"Task: {task}\nRelevant notes on hand: " + (
                ", ".join(n["label"] for n in top) if top else "none"
            )
            try:
                plan = call_gemini(cfg, AGENT_PLAN_SYSTEM, [{"role": "user", "content": context}])
            except Exception as err:  # noqa: BLE001
                print(f"[agent] plan failed: {err}")
                self._send_json(502, {"error": "Could not draft a plan, sir."})
                return
            session["pending"] = task
            self._send_json(200, {"answer": plan, "pending": True})
            return

        # stage 2: "do it" — execute the stored pending task
        pending = session.pop("pending", None)
        if not pending:
            self._send_json(409, {"error": "Nothing awaits confirmation, sir — give me a task first."})
            return
        top = score_notes(pending, self.notes_index)
        try:
            deliverable = call_gemini(
                cfg,
                AGENT_EXEC_SYSTEM.format(notes=format_notes(top)),
                [{"role": "user", "content": pending}],
                max_tokens=1024,  # deliverables run longer than spoken replies
            )
        except Exception as err:  # noqa: BLE001
            print(f"[agent] execution failed: {err}")
            self._send_json(502, {"error": "The task fell apart mid-flight, sir. Try again."})
            return

        # file the deliverable into the galaxy as a real capture
        result = remember_note(deliverable, self.notes_index, title=title_from_text(pending))
        self.__class__.notes_index = build_index(load_notes())
        log_activity("agent", pending, result["node"]["id"])
        result["answer"] = deliverable
        result["spoken"] = "Done, sir. The result is on screen and filed to the galaxy."
        print(f"[agent] executed '{pending[:60]}' -> node {result['node']['id']}")
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
        print(f"JARVIS online at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
