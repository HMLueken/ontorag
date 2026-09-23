"""Turning an LLM-supplied name into an IRI component.

Class, property and instance IRIs are built from names the model proposes, and a
model proposes what it reads: `BC1 RNA`, `α-synuclein`, `T-cell receptor (TCR)`.
A space in an IRI is not a serialisation nuisance, it is fatal — rdflib refuses
the whole graph:

    Exception: "http://…/scifact/BC1 RNA/59bf83a8" does not look like a valid URI

and it refuses it at *write* time, after every chunk has been extracted and paid
for. One unusual entity name in five thousand abstracts discards the run.

Both the schema TTL and the instance TTL derive IRIs from the same names, so they
have to derive them the *same way* — otherwise instances end up typed with a class
IRI the schema never defines, which no error reports and no query returns.

Whitespace becomes `_` because that is the common case and stays readable;
anything else outside the IRI-safe set is percent-encoded rather than dropped, so
two different names cannot collapse into one entity. The original name is kept as
`rdfs:label` by the callers, so nothing legible is lost.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import quote

_WS = re.compile(r"\s+")
# unreserved characters (RFC 3986 §2.3) plus the separators a name may legitimately carry
_SAFE = "-._~"


def local_name(name: str) -> str:
    """An IRI-safe local part for `name`, stable across runs and modules."""
    n = _WS.sub("_", (name or "").strip())
    if not n:
        # an unnamed class still needs an identity, and a stable one
        return "_" + hashlib.sha1((name or "").encode("utf-8")).hexdigest()[:8]
    return quote(n, safe=_SAFE)
