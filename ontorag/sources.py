"""Naming for source documents, shared by the graph writer and the dataset packager.

A document is a *pack* in the OntoRAG dataset format (one per source), and the same
slug has to appear in three places that are written by different code: the pack and
source IRIs in world.ttl, the chunk file name `content/chunks/<slug>.jsonl`, and the
key in `content/sources.json`. Deriving it in one place keeps them joinable.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def doc_slug(source_path: Optional[str], document_id: str) -> str:
    """Readable, stable pack slug for a document: its file stem, lower-cased and
    hyphenated, plus a short content-hash suffix so two files with the same name
    never share a pack."""
    stem = os.path.splitext(os.path.basename(source_path or ""))[0]
    base = _NON_SLUG.sub("-", stem.lower()).strip("-")
    digest = (document_id or "").removeprefix("doc_")[:6]
    if not base:
        return f"doc-{digest}" if digest else "doc"
    return f"{base}-{digest}" if digest else base


def doc_title(source_path: Optional[str]) -> str:
    """Human title for a document when nothing better is known: its file stem."""
    stem = os.path.splitext(os.path.basename(source_path or ""))[0]
    return stem.replace("_", " ").strip() or "Untitled document"


def file_sha256(path: Optional[str]) -> Optional[str]:
    """SHA-256 of a file's bytes, or None when the file is not reachable from here."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()
