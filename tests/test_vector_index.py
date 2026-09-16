"""Vector index: the logic offline, the database when the extension is present.

CI has no sqlite-vector binary, so everything that needs one is skipped rather
than faked — a fake would only prove the fake works. What is tested unconditionally
is the part that decides whether a rebuild happens, which is where the bug was:
`hash()` is randomised per process, so a stamp built from it never matched and the
cache never hit.
"""
import json
import struct

import pytest

from ontorag import vector_index as vi


def has_extension():
    try:
        vi.extension_path()
        return True
    except vi.VectorIndexError:
        return False


needs_ext = pytest.mark.skipif(
    not has_extension(),
    reason="sqlite-vector not present (ontorag index --download, or ONTORAG_VECTOR_EXTENSION)",
)


def make_dataset(tmp_path, n=40, dim=8, with_chunks=True):
    ds = tmp_path / "ds"
    (ds / "embeddings" / "vectors").mkdir(parents=True)
    (ds / "embeddings" / "config.json").write_text(json.dumps(
        {"provider": "test", "model": "unit", "dim": dim, "metric": "cosine", "normalized": True}
    ))
    # deterministic, distinguishable, unit-length vectors
    for doc in ("alpha", "beta"):
        lines = []
        for i in range(n // 2):
            v = [0.0] * dim
            v[(i + (0 if doc == "alpha" else 1)) % dim] = 1.0
            lines.append(json.dumps({"id": f"{doc}::{i:04d}", "vector": v}))
        (ds / "embeddings" / "vectors" / f"{doc}.jsonl").write_text("\n".join(lines) + "\n")
    if with_chunks:
        (ds / "content" / "chunks").mkdir(parents=True)
        for doc in ("alpha", "beta"):
            rows = [json.dumps({"id": f"{doc}::{i:04d}", "text": f"{doc} passage {i}"})
                    for i in range(n // 2)]
            (ds / "content" / "chunks" / f"{doc}.jsonl").write_text("\n".join(rows) + "\n")
    return ds


# ── offline ──────────────────────────────────────────────────────────

def test_stamp_is_stable_across_processes(tmp_path):
    """The bug this file exists for: a stamp must survive a restart."""
    ds = make_dataset(tmp_path)
    first = vi.read_embeddings(ds).stamp()
    assert first == vi.read_embeddings(ds).stamp()
    assert len(first) == 32 and all(c in "0123456789abcdef" for c in first)


def test_stamp_changes_when_a_shard_does(tmp_path):
    ds = make_dataset(tmp_path)
    before = vi.read_embeddings(ds).stamp()
    shard = ds / "embeddings" / "vectors" / "alpha.jsonl"
    shard.write_text(shard.read_text() + json.dumps(
        {"id": "alpha::9999", "vector": [1.0] + [0.0] * 7}) + "\n")
    assert vi.read_embeddings(ds).stamp() != before


def test_stamp_changes_when_the_embedding_contract_does(tmp_path):
    """A re-embed with another model must not be served from the old index."""
    ds = make_dataset(tmp_path)
    before = vi.read_embeddings(ds).stamp()
    cfg = ds / "embeddings" / "config.json"
    c = json.loads(cfg.read_text())
    c["model"] = "something-else"
    cfg.write_text(json.dumps(c))
    assert vi.read_embeddings(ds).stamp() != before


def test_missing_embeddings_say_so(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(vi.VectorIndexError, match="no embeddings/config.json"):
        vi.read_embeddings(tmp_path / "empty")


def test_unknown_quantisation_is_refused(tmp_path):
    ds = make_dataset(tmp_path)
    with pytest.raises(vi.VectorIndexError, match="unknown quantisation"):
        vi.build(ds, out=tmp_path / "x.db", quantize="turbo9")


def test_renamed_extension_is_caught_with_a_useful_message(tmp_path, monkeypatch):
    """SQLite derives the entry point from the file name; a renamed copy dies
    with `undefined symbol: sqlite3_<stem>_init`, which explains nothing."""
    fake = tmp_path / "vector-copy.so"
    fake.write_bytes(b"")
    monkeypatch.setenv("ONTORAG_VECTOR_EXTENSION", str(fake))
    with pytest.raises(vi.VectorIndexError, match="must be named vector"):
        vi.extension_path()


def test_asset_names_match_the_published_releases():
    import platform
    asset, member = vi._asset()
    assert vi.EXTENSION_VERSION in asset and asset.endswith(".tar.gz")
    assert member in ("vector.so", "vector.dylib", "vector.dll")
    assert platform.system().lower()[:3] in asset or "macos" in asset


def test_normalized_only_claimed_for_cosine():
    assert "normalized=1" in vi._init_sql(768, "cosine", True)
    # the flag is an assertion about unit length that only cosine can exploit
    assert "normalized=1" not in vi._init_sql(768, "l2", True)
    assert "normalized=1" not in vi._init_sql(768, "cosine", False)


# ── with the extension ───────────────────────────────────────────────

@needs_ext
def test_build_then_search_finds_the_query_itself(tmp_path):
    ds = make_dataset(tmp_path)
    out = tmp_path / "v.db"
    res = vi.build(ds, out=out, quantize="none")
    assert res.vectors == 40 and res.chunks == 40 and out.is_file()

    db = vi.open_db(out)
    v = [0.0] * 8
    v[3] = 1.0
    hits = vi.search(db, v, k=3)
    assert hits and hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)
    assert hits[0]["text"].endswith(hits[0]["chunk_id"].split("::")[1].lstrip("0") or "0")
    db.close()


@needs_ext
def test_unchanged_shards_are_not_rebuilt(tmp_path):
    ds = make_dataset(tmp_path)
    out = tmp_path / "v.db"
    vi.build(ds, out=out, quantize="none")
    again = vi.build(ds, out=out, quantize="none")
    assert again.skipped and again.vectors == 40
    assert not vi.build(ds, out=out, quantize="none", force=True).skipped


@needs_ext
def test_changed_shards_are_rebuilt(tmp_path):
    ds = make_dataset(tmp_path)
    out = tmp_path / "v.db"
    vi.build(ds, out=out, quantize="none")
    shard = ds / "embeddings" / "vectors" / "beta.jsonl"
    shard.write_text(shard.read_text() + json.dumps(
        {"id": "beta::9999", "vector": [0.0] * 7 + [1.0]}) + "\n")
    res = vi.build(ds, out=out, quantize="none")
    assert not res.skipped and res.vectors == 41


@needs_ext
def test_shipped_file_drops_the_vectors_and_still_answers(tmp_path):
    ds = make_dataset(tmp_path)
    out = tmp_path / "ship.db"
    vi.build(ds, out=out, quantize="turbo4", with_chunks=False, ship=True)
    db = vi.open_db(out)
    assert db.execute("SELECT length(embedding) FROM chunk_vectors LIMIT 1").fetchone()[0] == 0
    v = [0.0] * 8
    v[2] = 1.0
    assert vi.search(db, v, k=3), "a shipped index must still answer"
    db.close()


@needs_ext
def test_ship_without_quantisation_is_refused(tmp_path):
    ds = make_dataset(tmp_path)
    with pytest.raises(vi.VectorIndexError, match="--ship needs a quantisation"):
        vi.build(ds, out=tmp_path / "x.db", quantize="none", ship=True)


@needs_ext
def test_a_wrong_width_query_is_refused(tmp_path):
    ds = make_dataset(tmp_path)
    out = tmp_path / "v.db"
    vi.build(ds, out=out, quantize="none")
    db = vi.open_db(out)
    with pytest.raises(vi.VectorIndexError, match="dims"):
        vi.search(db, [0.1, 0.2], k=1)
    db.close()


@needs_ext
def test_a_shard_that_disagrees_with_the_config_is_refused(tmp_path):
    ds = make_dataset(tmp_path)
    shard = ds / "embeddings" / "vectors" / "alpha.jsonl"
    shard.write_text(json.dumps({"id": "alpha::bad", "vector": [1.0, 2.0]}) + "\n")
    with pytest.raises(vi.VectorIndexError, match="dims"):
        vi.build(ds, out=tmp_path / "x.db", quantize="none")


@needs_ext
def test_the_cache_ignores_itself(tmp_path):
    """The index lands in a directory the dataset should never commit."""
    ds = make_dataset(tmp_path)
    vi.build(ds, quantize="none")
    assert (ds / ".ontorag" / "vectors.db").is_file()
    assert "*" in (ds / ".ontorag" / ".gitignore").read_text()


@needs_ext
def test_blob_query_vectors_are_accepted(tmp_path):
    ds = make_dataset(tmp_path)
    out = tmp_path / "v.db"
    vi.build(ds, out=out, quantize="none")
    db = vi.open_db(out)
    v = [0.0] * 8
    v[1] = 1.0
    assert vi.search(db, struct.pack("<8f", *v), k=2)
    db.close()
