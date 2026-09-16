"""Materialise a dataset's vectors into a sqlite-vector database.

The committed form stays what it is: `embeddings/vectors/<doc>.jsonl`, one line
per chunk, diffable, and incremental in the same way extraction is — a new
document is a new file, not a rewritten blob. This builds the *query* form from
it, and the query form is a cache: derived, disposable, and rebuilt when the
shards change.

Why bother: a consumer that wants nearest-neighbour search over a dataset today
has to read every shard and hold every vector in memory first — 144 MB of JSONL
and ~555 MB of Python objects for a 17,942-chunk corpus, before it can answer
anything. The same vectors in SQLite, with sqlite-vector's TurboQuant index, open
in 46 ms, answer in ~19 ms, and cost 27 MB of RSS.

Two artefacts come out of here:

* **the cache** (`.ontorag/vectors.db`) — full-precision vectors plus the
  quantised index, written next to the dataset and never committed.
* **the shippable file** (`--ship`) — the quantised index with the
  full-precision column emptied, ~9 MB where the shards are 144 MB, or ~33 MB
  with the chunk text carried along so the file answers on its own. Small enough
  to attach to a release, which is a way to publish vectors that a private repo
  can share without putting a binary in its git history.

sqlite-vector is an extension, not a library: `vector_init` must be called on
**every connection** that queries the table. `open_db()` does that; a consumer in
another language must do the same.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import struct
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ontorag.verbosity import get_logger

_log = get_logger("ontorag.vector_index")

# Pinned: the extension is a binary we fetch, so the version is part of the
# build, not "whatever is newest today".
EXTENSION_VERSION = "1.1.0"
EXTENSION_REPO = "sqliteai/sqlite-vector"

TABLE = "chunk_vectors"
COLUMN = "embedding"
DEFAULT_REL = ".ontorag/vectors.db"

QUANTIZERS = {
    # name        (qtype,   qbits)  — measured on a 17,942 × 768 corpus:
    "turbo4": ("TURBO", 4),  # 7.1 MB index, 19 ms, recall@10 94.8%
    "turbo3": ("TURBO", 3),  # 5.4 MB index, recall@10 92.0%
    "turbo2": ("TURBO", 2),  # 3.7 MB index, 11 ms, recall@10 85.2%
    "int8": ("INT8", None),  # 13.9 MB index, 15 ms, recall@10 98.8%
    "1bit": ("1BIT", None),  # 1.9 MB index, 0.7 ms, recall@10 70.0%
    "none": (None, None),  # exact scan only: 65 ms, no index
}


class VectorIndexError(RuntimeError):
    """Something the user can act on: no extension, no embeddings, bad config."""


# ── the extension binary ─────────────────────────────────────────────


def _asset() -> tuple[str, str]:
    """(release asset name, file name inside it) for this machine."""
    m = platform.machine().lower()
    arm = m in ("arm64", "aarch64")
    system = platform.system()
    if system == "Linux":
        return f"vector-linux-{'arm64' if arm else 'x86_64'}-{EXTENSION_VERSION}.tar.gz", "vector.so"
    if system == "Darwin":
        return f"vector-macos-{'arm64' if arm else 'x86_64'}-{EXTENSION_VERSION}.tar.gz", "vector.dylib"
    if system == "Windows":
        return f"vector-windows-x86_64-{EXTENSION_VERSION}.tar.gz", "vector.dll"
    raise VectorIndexError(f"no sqlite-vector build for {system}/{m}")


def cache_dir() -> Path:
    root = os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(root) / "ontorag" / "sqlite-vector" / EXTENSION_VERSION


def extension_path(download: bool = False) -> Path:
    """Where the extension is, fetching it once if allowed.

    Order: an explicit path in `ONTORAG_VECTOR_EXTENSION`, then the cache. There
    is no PyPI package to depend on — the `sqlite-vector` name on PyPI is an
    unrelated, abandoned project — so the binary comes from the upstream release
    or from the caller.
    """
    override = os.getenv("ONTORAG_VECTOR_EXTENSION")
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise VectorIndexError(f"ONTORAG_VECTOR_EXTENSION={p} is not a file")
        # SQLite derives the entry point from the file name (sqlite3_<stem>_init),
        # so a renamed copy fails with an undefined symbol rather than "wrong file"
        if p.stem != "vector":
            raise VectorIndexError(
                f"{p.name} must be named vector{p.suffix} — SQLite takes the "
                f"extension's entry point from its file name"
            )
        return p

    asset, member = _asset()
    local = cache_dir() / member
    if local.is_file():
        return local
    if not download:
        raise VectorIndexError(
            f"sqlite-vector {EXTENSION_VERSION} not found. Fetch it once with "
            f"`ontorag index --download`, or set ONTORAG_VECTOR_EXTENSION to a "
            f"copy of {member} from https://github.com/{EXTENSION_REPO}/releases"
        )

    url = f"https://github.com/{EXTENSION_REPO}/releases/download/{EXTENSION_VERSION}/{asset}"
    _log.info("downloading sqlite-vector %s from %s", EXTENSION_VERSION, url)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.parent / (asset + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh:
            fh.write(r.read())
        with tarfile.open(tmp) as t:
            names = [n for n in t.getnames() if n.endswith(member)]
            if not names:
                raise VectorIndexError(f"{asset} does not contain {member}")
            src = t.extractfile(names[0])
            if src is None:
                raise VectorIndexError(f"{asset}: {names[0]} is not a file")
            local.write_bytes(src.read())
        local.chmod(0o755)
    finally:
        tmp.unlink(missing_ok=True)
    return local


# ── the database ─────────────────────────────────────────────────────


def connect(path: str | Path, extension: Optional[Path] = None) -> sqlite3.Connection:
    db = sqlite3.connect(str(path))
    db.enable_load_extension(True)
    # dlopen searches the library path for a bare name, so pass an explicit one
    db.load_extension(str((extension or extension_path()).resolve()))
    db.enable_load_extension(False)
    return db


def _init_sql(dim: int, metric: str, normalized: bool) -> str:
    opts = f"dimension={dim},type=FLOAT32,distance={metric.upper()}"
    if normalized and metric.lower() == "cosine":
        # an assertion, not a request: the shards are written L2-normalized, and
        # saying so lets a full-precision cosine scan compute 1 - dot instead
        opts += ",normalized=1"
    return opts


def open_db(path: str | Path, extension: Optional[Path] = None) -> sqlite3.Connection:
    """Open a built index ready to query — `vector_init` included.

    Every connection needs it; the call is metadata, not work.
    """
    db = connect(path, extension)
    row = db.execute("SELECT value FROM ontorag_meta WHERE key='vector_init'").fetchone()
    if not row:
        raise VectorIndexError(f"{path} is not an ontorag vector index")
    db.execute(f"SELECT vector_init('{TABLE}','{COLUMN}',?)", (row[0],))
    return db


# ── inputs ───────────────────────────────────────────────────────────


@dataclass
class Embeddings:
    config: dict
    shards: list[Path]

    @property
    def dim(self) -> int:
        return int(self.config["dim"])

    def stamp(self) -> str:
        """Fingerprint of the inputs, so a rebuild can be skipped and staleness
        detected. Name + size + mtime, not a content hash: 144 MB of JSONL is not
        worth re-reading to decide whether to re-read it.

        sha256, not `hash()`: Python randomises string hashing per process, so a
        built-in hash would differ on every run and the cache would never hit.
        """
        h = hashlib.sha256()
        h.update(json.dumps(self.config, sort_keys=True).encode())
        for s in self.shards:
            st = s.stat()
            h.update(f"\0{s.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        return h.hexdigest()[:32]


def read_embeddings(dataset: Path) -> Embeddings:
    cfg_path = dataset / "embeddings" / "config.json"
    if not cfg_path.is_file():
        raise VectorIndexError(
            f"{dataset} has no embeddings/config.json — this dataset carries no vectors"
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    shards = sorted((dataset / "embeddings" / "vectors").glob("*.jsonl"))
    if not shards:
        raise VectorIndexError(f"{dataset}/embeddings/vectors has no .jsonl shards")
    return Embeddings(cfg, shards)


def _rows(emb: Embeddings) -> Iterator[tuple[str, str, bytes]]:
    dim = emb.dim
    fmt = f"<{dim}f"
    for shard in emb.shards:
        doc = shard.stem
        with shard.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                v = r["vector"]
                if len(v) != dim:
                    raise VectorIndexError(
                        f"{shard.name}:{lineno} has {len(v)} dims, config.json says {dim}"
                    )
                yield r["id"], doc, struct.pack(fmt, *v)


def _chunk_rows(dataset: Path) -> Iterator[tuple[str, str, str]]:
    for shard in sorted((dataset / "content" / "chunks").glob("*.jsonl")):
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    c = json.loads(line)
                    yield c["id"], shard.stem, c.get("text", "")


# ── build ────────────────────────────────────────────────────────────


@dataclass
class BuildResult:
    path: Path
    vectors: int
    chunks: int
    quantized: int
    seconds: float
    bytes: int
    skipped: bool = False


def build(
    dataset: str | Path,
    out: Optional[str | Path] = None,
    quantize: str = "turbo4",
    with_chunks: bool = True,
    ship: bool = False,
    force: bool = False,
    extension: Optional[Path] = None,
) -> BuildResult:
    """Build (or refresh) the query form of a dataset's vectors."""
    dataset = Path(dataset)
    if quantize not in QUANTIZERS:
        raise VectorIndexError(
            f"unknown quantisation {quantize!r}; choose from {', '.join(QUANTIZERS)}"
        )
    emb = read_embeddings(dataset)
    out = Path(out) if out else dataset / DEFAULT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    # The dataset is a git repo and this file is a cache, so keep it out of the
    # working tree's way rather than relying on the dataset to know about us.
    if out.parent != dataset:
        gi = out.parent / ".gitignore"
        if not gi.exists():
            gi.write_text("# ontorag: derived vector index, never commit\n*\n", encoding="utf-8")

    stamp = emb.stamp()
    if out.is_file() and not force:
        try:
            db = connect(out, extension)
            row = db.execute("SELECT value FROM ontorag_meta WHERE key='stamp'").fetchone()
            n = db.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
            db.close()
            if row and row[0] == stamp:
                return BuildResult(out, n, 0, 0, 0.0, out.stat().st_size, skipped=True)
        except (sqlite3.DatabaseError, VectorIndexError):
            pass  # unreadable or older layout: rebuild

    t0 = time.time()
    tmp = out.with_suffix(out.suffix + ".building")
    tmp.unlink(missing_ok=True)
    db = connect(tmp, extension)
    db.executescript(
        f"""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE ontorag_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE {TABLE} (
          id        INTEGER PRIMARY KEY,
          chunk_id  TEXT NOT NULL UNIQUE,
          doc       TEXT NOT NULL,
          {COLUMN}  BLOB NOT NULL
        );
        """
    )

    n = 0
    batch: list[tuple[str, str, bytes]] = []
    insert = f"INSERT INTO {TABLE}(chunk_id,doc,{COLUMN}) VALUES (?,?,?)"
    for row in _rows(emb):
        batch.append(row)
        if len(batch) >= 2000:
            db.executemany(insert, batch)
            n += len(batch)
            batch = []
    if batch:
        db.executemany(insert, batch)
        n += len(batch)

    chunks = 0
    if with_chunks and (dataset / "content" / "chunks").is_dir():
        db.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc TEXT, text TEXT)")
        crows: list[tuple[str, str, str]] = []
        for row in _chunk_rows(dataset):
            crows.append(row)
            if len(crows) >= 2000:
                db.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?)", crows)
                chunks += len(crows)
                crows = []
        if crows:
            db.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?)", crows)
            chunks += len(crows)

    init = _init_sql(emb.dim, emb.config.get("metric", "cosine"),
                     bool(emb.config.get("normalized")))
    db.executemany(
        "INSERT OR REPLACE INTO ontorag_meta VALUES (?,?)",
        [
            ("stamp", stamp),
            ("vector_init", init),
            ("dim", str(emb.dim)),
            ("provider", str(emb.config.get("provider", ""))),
            ("model", str(emb.config.get("model", ""))),
            ("metric", str(emb.config.get("metric", "cosine"))),
            ("quantize", quantize),
            ("extension_version", EXTENSION_VERSION),
            ("shipped", "1" if ship else "0"),
        ],
    )
    db.commit()

    db.execute(f"SELECT vector_init('{TABLE}','{COLUMN}',?)", (init,))
    quantized = 0
    qtype, qbits = QUANTIZERS[quantize]
    if qtype:
        opts = f"qtype={qtype}" + (f",qbits={qbits}" if qbits else "")
        quantized = db.execute(
            f"SELECT vector_quantize('{TABLE}','{COLUMN}',?)", (opts,)
        ).fetchone()[0]
        db.commit()

    if ship:
        if not qtype:
            raise VectorIndexError("--ship needs a quantisation: the shipped file has no vectors")
        # The quantised index stands on its own, so the full-precision column is
        # dead weight in a file meant to be downloaded. Emptying it turns 82 MB
        # into 9 MB, and the scan still answers.
        db.execute(f"UPDATE {TABLE} SET {COLUMN} = zeroblob(0)")
        db.commit()
        db.execute("VACUUM")
    db.close()

    os.replace(tmp, out)
    return BuildResult(out, n, chunks, quantized, time.time() - t0, out.stat().st_size)


# ── query (the thing the whole file exists for) ──────────────────────


def search(db: sqlite3.Connection, vector: list[float] | bytes, k: int = 8,
           exact: bool = False) -> list[dict]:
    """Nearest chunks to a query vector.

    The query vector must come from the same provider+model as the dataset's —
    `ontorag_meta` records which, so a caller can check rather than guess.
    """
    if not isinstance(vector, (bytes, bytearray)):
        dim = int(db.execute("SELECT value FROM ontorag_meta WHERE key='dim'").fetchone()[0])
        if len(vector) != dim:
            raise VectorIndexError(f"query has {len(vector)} dims, index has {dim}")
        vector = struct.pack(f"<{dim}f", *vector)

    quantized = db.execute(
        "SELECT value FROM ontorag_meta WHERE key='quantize'").fetchone()
    fn = "vector_full_scan" if (exact or not quantized or quantized[0] == "none") \
        else "vector_quantize_scan"
    has_text = db.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()[0]

    # One query, whatever the consumer wants back: the join is why the file
    # exists rather than a pile of shards.
    sql = f"""
        SELECT v.chunk_id, v.doc, s.distance{", c.text" if has_text else ""}
        FROM {fn}('{TABLE}','{COLUMN}',?,?) AS s
        JOIN {TABLE} v ON v.id = s.rowid
        {"LEFT JOIN chunks c ON c.chunk_id = v.chunk_id" if has_text else ""}
        ORDER BY s.distance
    """
    out = []
    for row in db.execute(sql, (vector, k)).fetchall():
        hit = {"chunk_id": row[0], "doc": row[1], "distance": row[2]}
        if has_text:
            hit["text"] = row[3]
        out.append(hit)
    return out
