# JARVIS

An interactive 3D knowledge galaxy with a voice, a brain, and a butler's manners. Your markdown notes become stars in a force-directed galaxy; a dry, impeccably polite British assistant answers questions from them, grows the graph as you speak, briefs you on the weather, and remembers what you were doing last Tuesday.

Built with the Python standard library and a single HTML page — no pip installs, no npm, no build tools.

## Features

- **3D knowledge galaxy** — every `.md` note is a glowing star, color-coded by folder, linked by title mentions and shared `[[wikilinks]]`. Cinematic starfield, bloom, idle camera drift.
- **Ask your notes** — type or speak a question; JARVIS retrieves the most relevant notes, answers in one witty sentence plus the facts (Gemini), then flies the camera to the source note and opens it as proof. Answers drawing on 4+ notes light up the whole cluster instead.
- **Voice, both directions** — Web Speech API output (prefers a British voice, naturally) and `webkitSpeechRecognition` input with a wake word: say "Jarvis…" to interrupt him mid-sentence (true barge-in), or "Jarvis, brief me" in one breath.
- **Grow the brain by voice** — "remember that…" writes a real markdown note into `notes/captures/`, births the star live at its most related node with a glow pulse, and flies to it.
- **Briefings** — "good morning" / "brief me" fetches live weather from Open-Meteo (free, no key) for your city plus the state of your second brain.
- **Time Machine** — "what was I doing last Tuesday?" replays your activity log for any day.
- **Agent hands** — "research…", "draft…", "summarize…" produces a plan first and acts only after you say "do it". Deliverables are filed into the galaxy as new notes.

## Quick start

```bash
# 1. Build the graph from your notes
python3 build.py

# 2. Add your Gemini API key (either option)
#    a) paste it into config.json, or
#    b) export GEMINI_API_KEY=...  (also read from a root .env file)

# 3. Serve
python3 server.py
# open http://localhost:4700
```

Put your own notes in `notes/` (any folder structure — folders become groups) and re-run `build.py`.

## Deploy (Render, free)

This repo includes a `render.yaml` blueprint — one service runs everything (the Python server serves both the API and the viewer).

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect this GitHub repo (`JARVIS-AI`). Render reads `render.yaml` automatically.
3. When prompted, paste your `GEMINI_API_KEY`.
4. Deploy. Your JARVIS is live at `https://jarvis-ai-<something>.onrender.com`.

Notes on the free tier: the service sleeps after ~15 minutes idle (first request of the day takes ~50s to wake), and the disk is ephemeral — "remember that…" captures and the Time Machine log reset on redeploys. The 25 built-in notes always survive, since they're baked in at build time.

## Architecture

```
notes/               your markdown notes (25 sample notes included)
build.py             stdlib-only scanner -> viewer/graph-data.js
server.py            stdlib-only HTTP server, port 4700
  GET  /             serves viewer/ ONLY (config, .env, logs unreachable)
  POST /chat         retrieval + Gemini, per-session history
  POST /remember     writes a capture note, returns the new node + links
  GET  /briefing     Open-Meteo weather + second-brain status
  POST /settings     geocodes and stores your city (server-side only)
  POST /timemachine  summarizes your activity log for a given day
  POST /agent        plan -> "do it" confirmation -> execute + file result
viewer/index.html    the entire frontend (3d-force-graph via CDN)
```

## Security notes

- The API key lives in `config.json` / `.env` at the project root — the server only ever serves `viewer/`, so keys are unreachable from the browser (path traversal is blocked too).
- `settings.json` (your city) and `jarvis-log.jsonl` (your activity) are gitignored.
- The agent never acts without an explicit "do it".

## Requirements

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/apikey) for chat, briefings, and the agent (the galaxy itself works without one)
- Chrome for voice input (wake word + mic); speech output works in any modern browser
