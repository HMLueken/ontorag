from __future__ import annotations
import re
from typing import Optional, Dict, Any

from fastmcp import FastMCP
from pydantic import BaseModel

from ontorag.mcp_backend import GraphBackend, SparqlBackend
from ontorag.verbosity import get_logger

_log = get_logger("ontorag.mcp_server")


def _sanitize_iri(iri: str) -> str:
    """Reject IRIs that could break SPARQL angle-bracket syntax."""
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.\-]*:', iri):
        raise ValueError(f"Invalid IRI scheme: {iri!r}")
    if '>' in iri or '<' in iri:
        raise ValueError(f"IRI contains invalid characters: {iri!r}")
    return iri


def serve(app: FastMCP, host: str, port: int) -> None:
    """Serve an MCP app over HTTP on host:port.

    The transport must be named. FastMCP defaults to **stdio** and forwards any
    extra keyword arguments to the transport it picked, so `app.run(host=…,
    port=…)` — which is what this did — reaches `run_stdio_async()`, which takes
    neither, and the server dies at startup with

        TypeError: run_stdio_async() got an unexpected keyword argument 'host'

    on every fastmcp that has shipped a transport default (2.x, 3.x, 4.x alike).
    A CLI that offers --host and --port has to say http.
    """
    _log.info("Starting MCP server on %s:%d", host, port)
    app.run(transport="http", host=host, port=port)


def create_mcp_app(backend: GraphBackend) -> FastMCP:
    """Expose a graph backend over MCP.

    The four navigation tools work on any `GraphBackend`; the query-language tools
    are registered only when the backend actually speaks that language — SPARQL for
    RDF stores, Cypher for Neo4j — so a client never sees a tool that must fail.
    """
    _log.info("Creating MCP app with backend %s", type(backend).__name__)
    app = FastMCP("ontorag-mcp")

    @app.tool()
    def describe(iri: str, accept: str = "text/turtle") -> Dict[str, Any]:
        """DESCRIBE a resource by IRI."""
        _log.debug("tool:describe iri=%s", iri)
        iri = _sanitize_iri(iri)
        return {"content_type": accept, "data": backend.describe(iri, accept=accept)}

    @app.tool()
    def list_by_class(class_iri: str, limit: int = 50) -> Dict[str, Any]:
        """List instances of a class."""
        return backend.list_by_class(_sanitize_iri(class_iri), limit=limit)

    @app.tool()
    def outgoing(iri: str, limit: int = 100) -> Dict[str, Any]:
        """Outgoing edges from a resource."""
        return backend.outgoing(_sanitize_iri(iri), limit=limit)

    @app.tool()
    def incoming(iri: str, limit: int = 100) -> Dict[str, Any]:
        """Incoming edges to a resource."""
        return backend.incoming(_sanitize_iri(iri), limit=limit)

    @app.tool()
    def mentions(iris: list[str], limit: int = 8) -> Dict[str, Any]:
        """Source passages the given instances were extracted from (the citations)."""
        return backend.mentions([_sanitize_iri(i) for i in iris], limit=limit)

    if isinstance(backend, SparqlBackend):

        @app.tool()
        def sparql_select(query: str) -> Dict[str, Any]:
            """Run a SPARQL SELECT/ASK query and return SPARQL Results JSON."""
            _log.debug("tool:sparql_select query=%d chars", len(query))
            return backend.select(query)

        @app.tool()
        def sparql_construct(query: str, accept: str = "text/turtle") -> Dict[str, Any]:
            """Run a SPARQL CONSTRUCT/DESCRIBE and return RDF as text."""
            _log.debug("tool:sparql_construct query=%d chars accept=%s", len(query), accept)
            data = backend.construct(query, accept=accept)
            return {"content_type": accept, "data": data}

    elif hasattr(backend, "cypher"):

        @app.tool()
        def cypher_query(query: str, limit: int = 200) -> Dict[str, Any]:
            """Run a read-only Cypher query; rows are returned as SPARQL Results JSON."""
            _log.debug("tool:cypher_query query=%d chars", len(query))
            return backend.cypher(query, limit=limit)

    return app
