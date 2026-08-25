"""Offline tests for the Neo4j projection — no driver, no server.

The fixture graph is built with the real emitter (`instance_proposals_to_graph`),
so the mapping is tested against what extract-instances actually writes rather
than a hand-rolled approximation.
"""
import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from ontorag import neo4j_store as ns
from ontorag.instances_to_ttl import instance_proposals_to_graph

try:
    import neo4j  # noqa: F401
    HAVE_NEO4J = True
except ImportError:
    HAVE_NEO4J = False

NS = "http://amol/"
CHUNKS = {"c1": {"chunk_id": "c1", "provenance": {
    "source_path": "amol://core.pdf", "page": 12, "section": "Magi"}}}
PROPOSALS = [{
    "chunk_id": "c1",
    "instances": [{
        "class": "Magus",
        "label": "Bonisagus",
        "attributes": {"age": "1080", "house": "Bonisagus"},
        "relations": [{"predicate": "memberOf", "target_class": "Faction",
                       "target_label": "Order of Hermes"}],
        "mentions": [{"quote": "Bonisagus founded the Order."}],
    }],
}]


@pytest.fixture(scope="module")
def rows():
    g = instance_proposals_to_graph(CHUNKS, PROPOSALS, namespace=NS)
    return ns.graph_to_rows(g)


# ── naming ───────────────────────────────────────────────────────────

def test_label_and_rel_type_conventions():
    assert ns.safe_label("Character") == "Character"
    assert ns.safe_label("2Fast") == "N2Fast"          # labels can't start with a digit
    assert ns.rel_type("memberOf") == "MEMBER_OF"
    assert ns.rel_type("hasTag") == "HAS_TAG"
    assert ns.rel_type("capabilityDefinedInRuleSet") == "CAPABILITY_DEFINED_IN_RULE_SET"


def test_local_name_handles_both_iri_shapes():
    assert ns.local_name("https://rpg-schema.org/ns/rpg#Character") == "Character"
    assert ns.local_name("http://amol/Magus") == "Magus"


# ── projection ───────────────────────────────────────────────────────

def test_instances_become_labelled_nodes(rows):
    by_class = {n["classes"][0]: n for n in rows["nodes"]}
    assert set(by_class) == {"Magus", "Faction"}

    magus = by_class["Magus"]
    assert magus["label"] == "Bonisagus"
    assert magus["props"]["age"] == "1080"
    assert magus["props"]["house"] == "Bonisagus"
    # the full class IRI is kept, since local names collide across namespaces
    assert magus["class_iris"] == ["http://amol/Magus"]
    assert magus["iri"].startswith("http://amol/Magus/")


def test_object_property_becomes_a_relationship(rows):
    assert len(rows["rels"]) == 1
    rel = rows["rels"][0]
    assert rel["type"] == "MEMBER_OF"
    assert rel["iri"] == "http://amol/memberOf"
    iris = {n["iri"]: n["classes"][0] for n in rows["nodes"]}
    assert iris[rel["from"]] == "Magus" and iris[rel["to"]] == "Faction"


def test_mention_carries_the_citation_payload(rows):
    assert len(rows["mentions"]) == 1
    m = rows["mentions"][0]["props"]
    assert m["quote"] == "Bonisagus founded the Order."
    assert m["chunkId"] == "c1"
    assert m["sourcePath"] == "amol://core.pdf"
    assert m["page"] == 12                      # xsd:integer survives as an int
    assert m["section"] == "Magi"
    # attached to the instance it was derived from
    assert rows["mentions"][0]["iri"].startswith("http://amol/Magus/")


def test_reimport_is_idempotent(rows):
    """Mentions are blank nodes with no stable identity — the minted key is what
    stops a second import from duplicating every citation."""
    again = ns.graph_to_rows(instance_proposals_to_graph(CHUNKS, PROPOSALS, namespace=NS))
    assert [m["key"] for m in again["mentions"]] == [m["key"] for m in rows["mentions"]]
    assert [n["iri"] for n in again["nodes"]] == [n["iri"] for n in rows["nodes"]]


def test_mention_key_varies_with_quote_and_instance():
    a = ns.mention_key("http://amol/Magus/1", "c1", "quote one")
    assert a != ns.mention_key("http://amol/Magus/1", "c1", "quote two")
    assert a != ns.mention_key("http://amol/Magus/2", "c1", "quote one")


# ── edge cases ───────────────────────────────────────────────────────

def test_schema_triples_are_skipped():
    """Pointing at a concatenated schema+world TTL still yields only instances."""
    g = instance_proposals_to_graph(CHUNKS, PROPOSALS, namespace=NS)
    g.add((URIRef(NS + "Magus"), RDF.type, OWL.Class))
    g.add((URIRef(NS + "memberOf"), RDF.type, OWL.ObjectProperty))
    out = ns.graph_to_rows(g)
    assert {n["classes"][0] for n in out["nodes"]} == {"Magus", "Faction"}
    assert NS + "Magus" not in {n["iri"] for n in out["nodes"]}


def test_repeated_predicate_becomes_a_list():
    g = Graph()
    s = URIRef(NS + "Thing/1")
    g.add((s, RDF.type, URIRef(NS + "Thing")))
    g.add((s, URIRef(NS + "alias"), Literal("A")))
    g.add((s, URIRef(NS + "alias"), Literal("B")))
    props = ns.graph_to_rows(g)["nodes"][0]["props"]
    assert sorted(props["alias"]) == ["A", "B"]


def test_external_target_becomes_an_iri_property():
    """Edges are MATCHed on both ends at load time, so a target that is not an
    instance has nothing to match — but dc:license -> creativecommons.org is real
    data, so it is kept as a property instead of dropped (regression: the first cut
    silently lost 3 edges from the published AMOL graph)."""
    g = Graph()
    s = URIRef(NS + "RuleSet/1")
    g.add((s, RDF.type, URIRef(NS + "RuleSet")))
    g.add((s, URIRef("http://purl.org/dc/terms/license"),
           URIRef("https://creativecommons.org/licenses/by-sa/4.0/")))
    out = ns.graph_to_rows(g)
    assert out["rels"] == []
    assert out["nodes"][0]["props"]["license"] == "https://creativecommons.org/licenses/by-sa/4.0/"


def test_label_is_not_duplicated_into_props():
    g = Graph()
    s = URIRef(NS + "Thing/1")
    g.add((s, RDF.type, URIRef(NS + "Thing")))
    g.add((s, RDFS.label, Literal("Name")))
    node = ns.graph_to_rows(g)["nodes"][0]
    assert node["label"] == "Name" and "label" not in node["props"]


@pytest.mark.skipif(HAVE_NEO4J, reason="neo4j driver is installed")
def test_missing_driver_gives_an_install_hint():
    with pytest.raises(RuntimeError, match=r"ontorag\[neo4j\]"):
        ns._driver(None, None, None)
