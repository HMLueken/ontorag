"""hub push: manifest synthesis from a graph + corpus include/exclude selection.
Network (repo create / commit) is not exercised here — only the pure logic."""
from pathlib import Path

from ontorag.hub_push import _gather, _infer_graph_stats, _is_complete


_WORLD_TTL = """\
@prefix amol: <https://ontorag.dev/amol/> .
@prefix rpg:  <https://rpg-schema.org/ns/rpg#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

amol:Bonisagus a rpg:Character ; rdfs:label "Bonisagus" .
amol:Tremere   a rpg:Character ; rdfs:label "Tremere" .
amol:PilumOfFire a rpg:Spell ; rdfs:label "Pilum of Fire" .
"""


def _mk_dataset(tmp_path: Path) -> Path:
    (tmp_path / "ontology").mkdir()
    (tmp_path / "ontology/world.ttl").write_text(_WORLD_TTL, encoding="utf-8")
    (tmp_path / "content/sources").mkdir(parents=True)
    (tmp_path / "content/sources/book.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "content/sources.json").write_text('{"book.pdf": {"bytes": 12}}', encoding="utf-8")
    (tmp_path / "content/chunks.jsonl").write_text('{"chunk_id":"c1"}\n{"chunk_id":"c2"}\n', encoding="utf-8")
    (tmp_path / "content/dto").mkdir()
    (tmp_path / "content/dto/inter.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=secret", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/x.pyc").write_bytes(b"\x00")
    return tmp_path


def test_infer_graph_stats(tmp_path):
    g = tmp_path / "world.ttl"
    g.write_text(_WORLD_TTL, encoding="utf-8")
    base_iri, by_type, entities, prefixes = _infer_graph_stats(g)
    assert base_iri == "https://ontorag.dev/amol/"
    assert by_type == {"Character": 2, "Spell": 1}   # sorted by count desc
    assert entities == 3
    # only prefixes declared in the @prefix header AND used by the data — no
    # rdflib built-in noise (brick/csvw/…), no dc1 rebinding
    assert prefixes == {
        "amol": "https://ontorag.dev/amol/",
        "rpg": "https://rpg-schema.org/ns/rpg#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    }


def test_only_complete_manifests_are_published_as_is():
    """A Hub working manifest (no chunks_glob / entity_index) gets completed into the
    dataset format before publishing; a complete one is left alone."""
    working = {"ontorag": "0.1", "dataset": {"name": "X", "base_iri": "https://x/"},
               "content": {"chunks": "content/chunks.jsonl"}}
    assert not _is_complete(working)
    complete = {"ontorag": "0.1", "dataset": {"id": "x", "name": "X", "version": "0.1.0"},
                "ontology": {"graph": "ontology/world.ttl", "entity_index": "ontology/entities.jsonl",
                             "base_iri": "https://x/"},
                "content": {"chunks_glob": "content/chunks/*.jsonl"}}
    assert _is_complete(complete)


def test_gather_includes_corpus(tmp_path):
    d = _mk_dataset(tmp_path)
    files = _gather(d, include_sources=True)
    assert "content/sources/book.pdf" in files
    assert "ontology/world.ttl" in files
    assert "content/sources.json" in files
    # always-excluded noise
    assert ".env" not in files
    assert not any(p.startswith("__pycache__") for p in files)
    assert "content/dto/inter.jsonl" not in files


def test_gather_excludes_corpus_keeps_metadata(tmp_path):
    d = _mk_dataset(tmp_path)
    files = _gather(d, include_sources=False)
    assert "content/sources/book.pdf" not in files      # raw corpus dropped
    assert "content/sources.json" in files              # metadata kept
    assert "content/chunks.jsonl" in files               # derived content kept
    assert "ontology/world.ttl" in files
