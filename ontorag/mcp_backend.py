# ontorag/mcp_backend.py
"""Backends behind the knowledge-graph MCP tools.

Two layers:

* `GraphBackend` — the store-agnostic navigation the MCP tools actually need
  (`describe`, `list_by_class`, `outgoing`, `incoming`). Any store can implement it.
* `SparqlBackend` — a `GraphBackend` that also answers raw SPARQL. The navigation
  methods have a default SPARQL implementation here, so an RDF store only has to
  provide `select()` and `construct()`.

`Neo4jBackend` (in `neo4j_store.py`) implements `GraphBackend` directly in Cypher,
because Neo4j is an LPG and does not answer SPARQL. Both return **SPARQL Results
JSON** from the navigation methods, so the MCP tool contract is identical whichever
store is behind it.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import requests
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult

from ontorag.verbosity import get_logger

_log = get_logger("ontorag.mcp_backend")

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
PROV_DERIVED_FROM = "http://www.w3.org/ns/prov#wasDerivedFrom"
PROV_VALUE = "http://www.w3.org/ns/prov#value"


# ── SPARQL Results JSON helpers (the shared wire format) ─────────────

def term(value: Any, kind: str = "literal") -> Optional[Dict[str, str]]:
    """One SPARQL Results JSON binding cell. None drops the cell (unbound)."""
    if value is None:
        return None
    return {"type": kind, "value": str(value)}


def results_json(variables: Sequence[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble SPARQL Results JSON from rows of {var: cell-or-None}."""
    return {
        "head": {"vars": list(variables)},
        "results": {"bindings": [{k: v for k, v in row.items() if v is not None}
                                 for row in rows]},
    }


# ── interfaces ───────────────────────────────────────────────────────

class GraphBackend(ABC):
    """Store-agnostic graph navigation used by the MCP tools."""

    @abstractmethod
    def describe(self, iri: str, accept: str = "text/turtle") -> str:
        """Return the resource and its immediate surroundings as serialised RDF."""

    @abstractmethod
    def list_by_class(self, class_iri: str, limit: int = 50) -> Dict[str, Any]:
        """Instances of a class, as SPARQL Results JSON (?s ?label)."""

    @abstractmethod
    def outgoing(self, iri: str, limit: int = 100) -> Dict[str, Any]:
        """Outgoing edges/attributes, as SPARQL Results JSON (?p ?o)."""

    @abstractmethod
    def incoming(self, iri: str, limit: int = 100) -> Dict[str, Any]:
        """Incoming edges, as SPARQL Results JSON (?s ?p)."""

    @abstractmethod
    def mentions(self, iris: Sequence[str], limit: int = 8) -> Dict[str, Any]:
        """Provenance for the given instances, as SPARQL Results JSON
        (?s ?quote ?source ?chunkId).

        This is the "answers with receipts" primitive: every extracted fact links
        back to the passage it came from. It lives on the backend because the
        `mcp:Mention` model is ontorag's, and because a consumer should not have to
        know whether it is walking RDF or an LPG to get a citation.
        """


class SparqlBackend(GraphBackend):
    """A GraphBackend over a SPARQL store: implement select() + construct() and the
    navigation methods come for free."""

    @abstractmethod
    def select(self, query: str) -> Dict[str, Any]:
        """Run a SELECT/ASK query, return SPARQL Results JSON (as dict)."""

    @abstractmethod
    def construct(self, query: str, accept: str = "text/turtle") -> str:
        """Run a CONSTRUCT/DESCRIBE query, return serialised RDF as text."""

    # -- navigation, in SPARQL (previously inlined in mcp_server.py) --

    def describe(self, iri: str, accept: str = "text/turtle") -> str:
        return self.construct(f"DESCRIBE <{iri}>", accept=accept)

    def list_by_class(self, class_iri: str, limit: int = 50) -> Dict[str, Any]:
        return self.select(
            f"SELECT ?s ?label WHERE {{\n"
            f"  ?s a <{class_iri}> .\n"
            f"  OPTIONAL {{ ?s <{RDFS_LABEL}> ?label }}\n"
            f"}} LIMIT {int(limit)}"
        )

    def outgoing(self, iri: str, limit: int = 100) -> Dict[str, Any]:
        return self.select(f"SELECT ?p ?o WHERE {{ <{iri}> ?p ?o }} LIMIT {int(limit)}")

    def incoming(self, iri: str, limit: int = 100) -> Dict[str, Any]:
        return self.select(f"SELECT ?s ?p WHERE {{ ?s ?p <{iri}> }} LIMIT {int(limit)}")

    def mentions(self, iris: Sequence[str], limit: int = 8) -> Dict[str, Any]:
        if not iris:
            return results_json(["s", "quote", "source", "chunkId"], [])
        values = " ".join(f"<{i}>" for i in iris)
        # sourcePath/chunkId live in a per-dataset `mcp:` namespace derived from the
        # base IRI, so match them by local name rather than hardcoding a prefix
        return self.select(
            f"SELECT ?s ?quote ?source ?chunkId WHERE {{\n"
            f"  VALUES ?s {{ {values} }}\n"
            f"  ?s <{PROV_DERIVED_FROM}> ?m .\n"
            f"  ?m <{PROV_VALUE}> ?quote .\n"
            f'  OPTIONAL {{ ?m ?sp ?source . FILTER(STRENDS(STR(?sp), "sourcePath")) }}\n'
            f'  OPTIONAL {{ ?m ?cp ?chunkId . FILTER(STRENDS(STR(?cp), "chunkId")) }}\n'
            f"}} LIMIT {int(limit)}"
        )


# ── RDF implementations ──────────────────────────────────────────────

class LocalRdfBackend(SparqlBackend):
    """In-memory rdflib backend loaded from local TTL files."""

    def __init__(self, ontology_ttl: str, instances_ttl: str) -> None:
        _log.info("LocalRdfBackend: loading onto=%s inst=%s", ontology_ttl, instances_ttl)
        self._graph = Graph()
        self._graph.parse(ontology_ttl, format="turtle")
        self._graph.parse(instances_ttl, format="turtle")
        _log.info("LocalRdfBackend: loaded %d triples", len(self._graph))

    def select(self, query: str) -> Dict[str, Any]:
        result: SPARQLResult = self._graph.query(query)
        raw = result.serialize(format="json")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def construct(self, query: str, accept: str = "text/turtle") -> str:
        result = self._graph.query(query)
        out_graph = result if isinstance(result, Graph) else getattr(result, "graph", None)
        if out_graph is None:
            raise RuntimeError("Query did not return a graph result")

        fmt_map = {
            "text/turtle": "turtle",
            "application/ld+json": "json-ld",
            "application/rdf+xml": "xml",
            "application/n-triples": "nt",
        }
        fmt = fmt_map.get(accept, "turtle")
        data = out_graph.serialize(format=fmt)
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return data


class RemoteSparqlBackend(SparqlBackend):
    """Backend that proxies queries to a remote SPARQL endpoint."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        _log.info("RemoteSparqlBackend: endpoint=%s", endpoint)

    def select(self, query: str) -> Dict[str, Any]:
        r = requests.post(
            self._endpoint,
            data={"query": query},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/sparql-results+json",
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def construct(self, query: str, accept: str = "text/turtle") -> str:
        r = requests.post(
            self._endpoint,
            data={"query": query},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": accept,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.text
