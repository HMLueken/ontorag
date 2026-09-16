"""The GraphBackend layer: navigation is store-agnostic, and the MCP tool surface
adapts to what the backend can actually do.

Offline — the RDF side runs against a real in-memory graph; the Neo4j side is
exercised through a fake driver so no server (and no driver package) is needed.
"""
import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from ontorag import mcp_backend as mb
from ontorag.instances_to_ttl import instance_proposals_to_graph
from ontorag.mcp_backend import GraphBackend, LocalRdfBackend, SparqlBackend

NS = "http://amol/"
CHUNKS = {"c1": {"chunk_id": "c1", "provenance": {"source_path": "amol://core.pdf"}}}
PROPOSALS = [{
    "chunk_id": "c1",
    "instances": [{
        "class": "Magus", "label": "Bonisagus",
        "attributes": {"age": "1080"},
        "relations": [{"predicate": "memberOf", "target_class": "Faction",
                       "target_label": "Order of Hermes"}],
        "mentions": [{"quote": "Bonisagus founded the Order."}],
    }],
}]


@pytest.fixture(scope="module")
def rdf_backend(tmp_path_factory):
    d = tmp_path_factory.mktemp("ttl")
    onto, inst = d / "schema.ttl", d / "world.ttl"
    onto.write_text("", encoding="utf-8")
    instance_proposals_to_graph(CHUNKS, PROPOSALS, namespace=NS).serialize(
        destination=str(inst), format="turtle")
    return LocalRdfBackend(str(onto), str(inst))


def _values(res, var):
    return [b[var]["value"] for b in res["results"]["bindings"] if var in b]


# ── the shared wire format ───────────────────────────────────────────

def test_results_json_shape_and_unbound_cells():
    out = mb.results_json(["s", "label"], [
        {"s": mb.term("http://x/1", "uri"), "label": mb.term("A")},
        {"s": mb.term("http://x/2", "uri"), "label": mb.term(None)},   # OPTIONAL miss
    ])
    assert out["head"]["vars"] == ["s", "label"]
    assert out["results"]["bindings"][0]["s"] == {"type": "uri", "value": "http://x/1"}
    assert "label" not in out["results"]["bindings"][1]


# ── SPARQL backends get navigation for free ──────────────────────────

def test_sparql_backend_implements_graph_backend(rdf_backend):
    assert isinstance(rdf_backend, GraphBackend)


def test_list_by_class(rdf_backend):
    res = rdf_backend.list_by_class(NS + "Magus")
    assert _values(res, "label") == ["Bonisagus"]


def test_outgoing_and_incoming(rdf_backend):
    magus = _values(rdf_backend.list_by_class(NS + "Magus"), "s")[0]
    preds = _values(rdf_backend.outgoing(magus), "p")
    assert NS + "age" in preds and NS + "memberOf" in preds

    faction = _values(rdf_backend.list_by_class(NS + "Faction"), "s")[0]
    assert magus in _values(rdf_backend.incoming(faction), "s")


def test_describe_returns_turtle(rdf_backend):
    magus = _values(rdf_backend.list_by_class(NS + "Magus"), "s")[0]
    assert "Bonisagus" in rdf_backend.describe(magus)


# ── the Neo4j backend, against a fake driver ─────────────────────────

MAGUS = NS + "Magus/aaa"

class _FakeDriver:
    """Answers the handful of Cypher shapes Neo4jBackend issues."""
    def __init__(self, rows_for): self._rows_for = rows_for
    def session(self, **_): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute_read(self, fn): return fn(self)
    def run(self, cypher, **params): return self._rows_for(cypher, params)
    def close(self): pass


def _fake_backend(monkeypatch, rows_for):
    from ontorag import neo4j_store
    monkeypatch.setattr(neo4j_store, "_driver", lambda *a, **k: _FakeDriver(rows_for))
    return neo4j_store.Neo4jBackend()


class _Rec(dict):
    def data(self): return dict(self)


NODE = _Rec(props={"iri": MAGUS, "label": "Bonisagus", "age": "1080",
                   "classIris": [NS + "Magus"], "propertyIris": [NS + "age"]},
            classes=[NS + "Magus"])


def _rows(cypher, params):
    if "properties(n)" in cypher:
        return [NODE]
    if "WHERE $cls IN n.classIris" in cypher:
        return [_Rec(s=MAGUS, label="Bonisagus")]
    if "-[rel]->(b)" in cypher:
        return [_Rec(p=NS + "memberOf", o=NS + "Faction/bbb", mkey=None, quote=None,
                     bprops={"iri": NS + "Faction/bbb"})]
    if "(a:Resource)-[rel]->(b:Resource" in cypher:
        return [_Rec(s=MAGUS, p=NS + "memberOf")]
    return []


def test_neo4j_backend_is_a_graph_backend_but_not_sparql(monkeypatch):
    b = _fake_backend(monkeypatch, _rows)
    assert isinstance(b, GraphBackend)
    assert not isinstance(b, SparqlBackend)   # Neo4j does not answer SPARQL


def test_neo4j_navigation_returns_sparql_results_json(monkeypatch):
    b = _fake_backend(monkeypatch, _rows)
    res = b.list_by_class(NS + "Magus")
    assert res["head"]["vars"] == ["s", "label"]
    assert _values(res, "s") == [MAGUS]
    assert res["results"]["bindings"][0]["s"]["type"] == "uri"


def test_neo4j_outgoing_recovers_predicate_iris(monkeypatch):
    """Property keys are local names in the LPG; propertyIris maps them back, so the
    output matches what the RDF backend reports rather than inventing an IRI."""
    b = _fake_backend(monkeypatch, _rows)
    preds = _values(b.outgoing(MAGUS), "p")
    assert NS + "age" in preds          # recovered, not "age"
    assert NS + "memberOf" in preds
    assert str(RDF.type) in preds and str(RDFS.label) in preds


def test_neo4j_describe_emits_parsable_rdf(monkeypatch):
    b = _fake_backend(monkeypatch, _rows)
    g = Graph()
    g.parse(data=b.describe(MAGUS), format="turtle")
    s = URIRef(MAGUS)
    assert (s, RDF.type, URIRef(NS + "Magus")) in g
    assert (s, RDFS.label, Literal("Bonisagus")) in g
    assert (s, URIRef(NS + "age"), Literal("1080")) in g
    assert (s, URIRef(NS + "memberOf"), URIRef(NS + "Faction/bbb")) in g


def test_neo4j_describe_rebuilds_provenance_with_real_iris(monkeypatch):
    """RDF -> Neo4j -> RDF must reproduce the original predicates, not invented ones:
    the citation walk (prov:wasDerivedFrom -> prov:value) has to survive the trip."""
    MCP = NS + "mcp/"
    def rows(cypher, params):
        if "properties(n)" in cypher:
            return [NODE]
        if "-[rel]->(b)" in cypher:
            return [_Rec(p="http://www.w3.org/ns/prov#wasDerivedFrom", o=None,
                         mkey="k1", quote="q",
                         bprops={"key": "k1", "quote": "Bonisagus founded the Order.",
                                 "chunkId": "c1", "sourcePath": "amol://core.pdf",
                                 "propertyIris": [MCP + "chunkId", MCP + "sourcePath",
                                                  "http://www.w3.org/ns/prov#value"]})]
        return []

    b = _fake_backend(monkeypatch, rows)
    ttl = b.describe(MAGUS)
    assert "urn:ontorag" not in ttl                     # nothing fabricated
    g = Graph(); g.parse(data=ttl, format="turtle")
    PROV = "http://www.w3.org/ns/prov#"
    mention = next(g.objects(URIRef(MAGUS), URIRef(PROV + "wasDerivedFrom")))
    assert (mention, URIRef(PROV + "value"),
            Literal("Bonisagus founded the Order.")) in g
    assert (mention, URIRef(MCP + "chunkId"), Literal("c1")) in g


def test_neo4j_describe_skips_properties_with_no_recorded_iri(monkeypatch):
    """A bare local name is not a valid predicate IRI — omit it rather than fake one."""
    node = _Rec(props={"iri": MAGUS, "label": "X", "mystery": "v",
                       "classIris": [NS + "Magus"], "propertyIris": []},
                classes=[NS + "Magus"])
    b = _fake_backend(monkeypatch, lambda c, p: [node] if "properties(n)" in c else [])
    g = Graph()
    g.parse(data=b.describe(MAGUS), format="turtle")
    assert "mystery" not in g.serialize(format="turtle")


# ── serving the tools ────────────────────────────────────────────────


def test_serve_names_an_http_transport():
    """`ontorag mcp-server --host --port` shipped broken because it did not.

    FastMCP defaults to stdio and forwards unrecognised keyword arguments to the
    transport it chose, so `app.run(host=…, port=…)` reaches `run_stdio_async()`,
    which takes neither, and the command dies at startup.
    """
    from ontorag.mcp_server import serve

    seen = {}

    class FakeApp:
        def run(self, *args, **kwargs):
            seen.update(kwargs)
            seen["positional"] = args

    serve(FakeApp(), "127.0.0.1", 9010)
    assert seen.get("transport") == "http", "a transport must be named"
    assert seen.get("host") == "127.0.0.1" and seen.get("port") == 9010


def test_serve_arguments_fit_the_installed_fastmcp():
    """The version-drift half: checked against whatever fastmcp is installed, so
    CI fails here rather than in a user's terminal when the API moves again."""
    import inspect

    from fastmcp import FastMCP

    params = inspect.signature(FastMCP.run_http_async).parameters
    for kw in ("transport", "host", "port"):
        assert kw in params, f"fastmcp's HTTP transport no longer takes {kw}"
    assert "transport" in inspect.signature(FastMCP.run).parameters


# ── MCP tool surface adapts to the backend ───────────────────────────

async def _tool_names(backend):
    from ontorag.mcp_server import create_mcp_app
    tools = await create_mcp_app(backend).list_tools()
    return {t.name for t in tools}


NAV = {"describe", "list_by_class", "outgoing", "incoming"}


@pytest.mark.asyncio
async def test_rdf_backend_exposes_sparql_tools(rdf_backend):
    names = await _tool_names(rdf_backend)
    assert NAV <= names
    assert {"sparql_select", "sparql_construct"} <= names
    assert "cypher_query" not in names


@pytest.mark.asyncio
async def test_neo4j_backend_swaps_sparql_for_cypher(monkeypatch):
    names = await _tool_names(_fake_backend(monkeypatch, _rows))
    assert NAV <= names
    assert "cypher_query" in names
    assert not {"sparql_select", "sparql_construct"} & names


# ── provenance is a backend primitive, not a caller's job ────────────

def test_sparql_mentions_walks_prov(rdf_backend):
    magus = _values(rdf_backend.list_by_class(NS + "Magus"), "s")[0]
    res = rdf_backend.mentions([magus])
    assert res["head"]["vars"] == ["s", "quote", "source", "chunkId"]
    assert _values(res, "quote") == ["Bonisagus founded the Order."]
    # the mcp: namespace is per-dataset, so these are matched by local name
    assert _values(res, "source") == ["amol://core.pdf"]
    assert _values(res, "chunkId") == ["c1"]


def test_mentions_of_nothing_is_an_empty_result(rdf_backend):
    res = rdf_backend.mentions([])
    assert res["results"]["bindings"] == []
    assert res["head"]["vars"] == ["s", "quote", "source", "chunkId"]


def test_neo4j_mentions_uses_parameters_not_interpolation(monkeypatch):
    seen = {}
    def rows(cypher, params):
        seen.update(params)
        return [_Rec(s=MAGUS, quote="Bonisagus founded the Order.",
                     source="amol://core.pdf", chunkId="c1")]
    b = _fake_backend(monkeypatch, rows)
    res = b.mentions([MAGUS])
    assert seen["iris"] == [MAGUS]          # bound as a parameter, not inlined
    assert _values(res, "quote") == ["Bonisagus founded the Order."]
    assert _values(res, "source") == ["amol://core.pdf"]
