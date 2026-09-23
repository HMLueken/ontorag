"""build-dataset: a pipeline directory becomes a dataset that follows the OntoRAG
dataset format 0.1 (https://ontorag.org/vocab/#format).

The fixture is built with the real emitter and real DTO-shaped chunks, and every
file written is validated against the published JSON Schemas (copied into
tests/fixtures/dataset_format_0.1)."""
import json
from pathlib import Path

import pytest
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

from ontorag.dataset_package import complete_dataset
from ontorag.instances_to_ttl import instance_proposals_to_graph
from ontorag.sources import doc_slug

jsonschema = pytest.importorskip("jsonschema")

ORP = Namespace("https://ontorag.org/provenance#")
OA = Namespace("http://www.w3.org/ns/oa#")
SCHEMAS = Path(__file__).parent / "fixtures" / "dataset_format_0.1"
NS = "https://example.org/srd/"

CHUNKS = [
    {"document_id": "doc_8974902d109d6e63", "chunk_id": "doc_8974902d109d6e63#p131#c0412",
     "chunk_index": 412, "text": "Fireball. A bright streak flashes from you…",
     "text_hash": "x", "provenance": {"source_path": "content/sources/SRD_CC_v5.2.1.pdf",
                                      "source_mime": "application/pdf", "page": 131,
                                      "page_label": "131", "section": "Fireball"}},
    {"document_id": "doc_8974902d109d6e63", "chunk_id": "doc_8974902d109d6e63#p130#c0411",
     "chunk_index": 411, "text": "Fear. You project a phantasmal image…",
     "text_hash": "y", "provenance": {"source_path": "content/sources/SRD_CC_v5.2.1.pdf",
                                      "page": 130, "page_label": "130"}},
]
PROPOSALS = [{"chunk_id": CHUNKS[0]["chunk_id"], "instances": [{
    "class": "Spell", "label": "Fireball", "attributes": {"level": "3"},
    "mentions": [{"quote": "A bright streak flashes from you"}]}]}]


def _validator(kind):
    schema = json.loads((SCHEMAS / f"{kind}.schema.json").read_text())
    return jsonschema.Draft7Validator(schema)


@pytest.fixture()
def dataset(tmp_path):
    (tmp_path / "content").mkdir()
    (tmp_path / "ontology").mkdir()
    with open(tmp_path / "content/chunks.jsonl", "w") as f:
        for c in CHUNKS:
            f.write(json.dumps(c) + "\n")
    by_id = {c["chunk_id"]: c for c in CHUNKS}
    instance_proposals_to_graph(by_id, PROPOSALS, namespace=NS).serialize(
        destination=str(tmp_path / "ontology/world.ttl"), format="turtle")
    # a Hub working manifest: must survive, extended rather than replaced
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"ontorag": "0.1", "dataset": {"slug": "srd", "name": "SRD", "base_iri": NS},
         "baselines": ["rpg"], "hub": {"state": "extracted"},
         "content": {"sources": "content/sources.json", "chunks": "content/chunks.jsonl"}}))
    return tmp_path


def test_every_published_file_follows_the_dataset_format(dataset):
    m = complete_dataset(dataset, license="CC-BY-4.0")
    assert not list(_validator("manifest").iter_errors(m))
    chunk_files = sorted((dataset / "content/chunks").glob("*.jsonl"))
    assert [p.stem for p in chunk_files] == [doc_slug("SRD_CC_v5.2.1.pdf", "doc_8974902d109d6e63")]
    for kind, paths in (("chunk", chunk_files), ("entity", [dataset / m["ontology"]["entity_index"]])):
        for p in paths:
            for line in p.read_text().splitlines():
                assert not list(_validator(kind).iter_errors(json.loads(line))), (kind, line)


def test_manifest_is_merged_not_replaced(dataset):
    m = complete_dataset(dataset)
    assert m["hub"] == {"state": "extracted"} and m["baselines"] == ["rpg"]
    assert m["dataset"]["id"] == "srd" and m["dataset"]["version"] == "0.1.0"
    assert m["ontology"]["base_iri"] == NS
    assert m["content"]["chunks_glob"] == "content/chunks/*.jsonl"
    assert m["content"]["counts"] == {"documents": 1, "chunks": 2}
    # only instances count: not sources, mentions, targets or selectors
    assert m["ontology"]["counts"] == {"entities": 1, "by_type": {"Spell": 1}}
    assert "embeddings" not in m                       # no vectors built yet
    assert m["composition"]["dependency"] == "orp:requires"


def test_chunks_keep_pages_and_link_their_entities(dataset):
    complete_dataset(dataset)
    rows = [json.loads(l) for p in (dataset / "content/chunks").glob("*.jsonl")
            for l in p.read_text().splitlines()]
    assert [r["seq"] for r in rows] == [411, 412]
    fireball = rows[1]
    assert fireball["page"] == 131 and fireball["page_label"] == "131"
    assert fireball["heading_path"] == ["Fireball"]
    assert len(fireball["entities"]) == 1 and "/Spell/" in fireball["entities"][0]
    assert "entities" not in rows[0]


def test_entities_record_their_attesting_source(dataset):
    complete_dataset(dataset)
    (rec,) = [json.loads(l) for l in (dataset / "ontology/entities.jsonl").read_text().splitlines()]
    assert rec["label"] == "Fireball"
    assert rec["attestedIn"] == [doc_slug("SRD_CC_v5.2.1.pdf", "doc_8974902d109d6e63")]


def test_rerunning_is_idempotent(dataset):
    first = complete_dataset(dataset)
    files = {p: p.read_bytes() for p in (dataset / "content").rglob("*.json*")}
    second = complete_dataset(dataset)
    first.pop("dataset")["built_at"], second.pop("dataset")["built_at"]
    assert first == second
    assert files == {p: p.read_bytes() for p in (dataset / "content").rglob("*.json*")}


def test_graph_carries_orp_provenance():
    by_id = {c["chunk_id"]: c for c in CHUNKS}
    g = instance_proposals_to_graph(by_id, PROPOSALS, namespace=NS)
    (spell,) = list(g.subjects(RDF.type, URIRef(NS + "Spell")))
    (mention,) = list(g.objects(spell, ORP.hasMention))
    assert (mention, RDF.type, ORP.Mention) in g
    assert (mention, ORP.mentions, spell) in g
    target = g.value(mention, OA.hasTarget)
    file_ = g.value(target, OA.hasSource)
    assert (file_, RDF.type, ORP.SourceFile) in g
    source = g.value(file_, ORP.fileOf)
    assert (spell, ORP.attestedIn, source) in g
    selectors = {g.value(s, RDF.type): s for s in g.objects(target, OA.hasSelector)}
    assert str(g.value(selectors[OA.TextQuoteSelector], OA.exact)) == "A bright streak flashes from you"
    assert g.value(selectors[ORP.PageSelector], ORP.pageStart).toPython() == 131
    assert str(g.value(selectors[ORP.PageSelector], ORP.pageLabel)) == "131"
    chunk = g.value(mention, ORP.inChunk)
    pack = g.value(chunk, ORP.inPack)
    assert (pack, ORP.packOf, source) in g and g.value(pack, ORP.namedGraph) is not None
    # no per-dataset mcp: namespace anymore
    assert not any("/mcp/" in str(t) for triple in g for t in triple)
