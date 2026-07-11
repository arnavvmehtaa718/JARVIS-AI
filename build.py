#!/usr/bin/env python3
"""
build.py — scan ./notes for .md files and emit viewer/graph-data.js

Standard library only. Produces:
    const GRAPH = { nodes: [...], links: [...] }

Each node:
    id      — numeric, equal to its index in the nodes array (IMPORTANT:
              later features depend on looking nodes up by index)
    label   — filename without extension
    group   — immediate parent folder name
    excerpt — ~700 characters of cleaned note text

Two notes are linked when:
    - one note's text mentions the other's title (case-insensitive), or
    - they share at least one [[wikilink]] target.
"""

import json
import re
from pathlib import Path

NOTES_DIR = Path(__file__).parent / "notes"
OUT_FILE = Path(__file__).parent / "viewer" / "graph-data.js"
EXCERPT_LEN = 700

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def clean_text(raw: str) -> str:
    """Strip markdown syntax down to readable plain text."""
    text = raw
    # Unwrap wikilinks: [[Target]] -> Target, [[Target|Alias]] -> Alias
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Markdown links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Drop heading lines entirely (the filename already provides the title)
    text = re.sub(r"^#{1,6}\s.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`>#]+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_excerpt(text: str, limit: int = EXCERPT_LEN) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Avoid slicing mid-word
    last_space = cut.rfind(" ")
    if last_space > limit * 0.6:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def main() -> None:
    if not NOTES_DIR.is_dir():
        raise SystemExit(f"Notes folder not found: {NOTES_DIR}")

    md_files = sorted(NOTES_DIR.rglob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found under {NOTES_DIR}")

    nodes = []
    raw_texts = []   # lowercased raw markdown per node
    wikilinks = []   # set of lowercased wikilink targets per node

    for path in md_files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        label = path.stem
        rel = path.relative_to(NOTES_DIR)
        group = rel.parts[0] if len(rel.parts) > 1 else "root"

        nodes.append({
            "id": len(nodes),  # numeric id == index in nodes array
            "label": label,
            "group": group,
            "excerpt": make_excerpt(clean_text(raw)),
        })
        raw_texts.append(raw.lower())
        wikilinks.append({m.strip().lower() for m in WIKILINK_RE.findall(raw)})

    # Build links.
    # Co-citation-only links (notes that merely share wikilink targets) explode
    # into cliques around hub notes, so require MIN_SHARED common targets when
    # there is no direct title mention. Set to 1 for the loosest interpretation.
    MIN_SHARED = 2
    links = set()
    n = len(nodes)
    for i in range(n):
        title_i = nodes[i]["label"].lower()
        for j in range(i + 1, n):
            title_j = nodes[j]["label"].lower()
            mentions = title_j in raw_texts[i] or title_i in raw_texts[j]
            shared = len(wikilinks[i] & wikilinks[j]) >= MIN_SHARED
            if mentions or shared:
                links.add((i, j))

    graph = {
        "nodes": nodes,
        "links": [{"source": s, "target": t} for s, t in sorted(links)],
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        "const GRAPH = " + json.dumps(graph, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )

    groups = sorted({node["group"] for node in nodes})
    print(f"Scanned {n} notes in {len(groups)} groups: {', '.join(groups)}")
    print(f"Built {len(graph['links'])} links")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
