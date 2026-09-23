"""Complete a dataset directory into the OntoRAG dataset format.

The pipeline leaves a dataset directory in its *working* shape: DTO chunks in
`content/chunks.jsonl`, the graph in `ontology/world.ttl`, and whatever manifest the
Hub or the user started with. Services such as ontorag-mcp read the *published*
shape defined at https://ontorag.org/vocab/#format (dataset format 0.1): per-pack
chunk files, an entity index, a source registry and a manifest pointing at them.

`complete_dataset()` derives the published files from the working ones and merges
the required manifest fields into the existing manifest. It never deletes keys, so
Hub-specific fields (`hub`, `baselines`, `dataset.base_iri`, …) survive, and it is
idempotent: running it again after another extraction just refreshes the files.
"""
from __future__ import annotations

import datetime
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ontorag.sources import doc_slug, doc_title
from ontorag.verbosity import get_logger

_log = get_logger("ontorag.dataset")

SPEC_VERSION = "0.1"
SPEC_BASE = "https://ontorag.org/vocab/dataset/0.1/"
CHUNKS_DIR = "content/chunks"
ENTITY_INDEX = "ontology/entities.jsonl"

_ORP = "https://ontorag.org/provenance#"
_DCTERMS = "http://purl.org/dc/terms/"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_OA = "http://www.w3.org/ns/oa#"
_NOT_ENTITIES = {
    "http://www.w3.org/2002/07/owl#Class",
    "http://www.w3.org/2002/07/owl#ObjectProperty",
    "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#AnnotationProperty",
    "http://www.w3.org/2002/07/owl#Ontology",
    "http://www.w3.org/2000/01/rdf-schema#Class",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
}


def _is_entity_type(t: str) -> bool:
    """Instance classes, as opposed to schema terms and provenance scaffolding
    (orp: sources, packs, mentions, selectors; oa: targets and selectors)."""
    return t not in _NOT_ENTITIES and not t.startswith((_ORP, _OA))


# ── graph statistics ─────────────────────────────────────────────────

def split_iri(uri: str) -> Tuple[str, str]:
    s = str(uri)
    i = max(s.rfind("#"), s.rfind("/"))
    return (s[:i + 1], s[i + 1:]) if i >= 0 else ("", s)


def infer_graph_stats(graph_path: Path) -> Tuple[str, Dict[str, int], int, Dict[str, str]]:
    """(base_iri, by_type_local_counts, total_entities, prefixes) from the graph."""
    from rdflib import Graph, RDF, URIRef

    g = Graph()
    g.parse(str(graph_path), format="turtle")

    by_type: Dict[str, int] = {}
    subjects = set()
    ns_counter: Counter = Counter()
    used_ns = set()
    for s, p, o in g:
        for term in (s, p, o):
            if isinstance(term, URIRef):
                used_ns.add(split_iri(str(term))[0])
        if (p == RDF.type and isinstance(s, URIRef) and isinstance(o, URIRef)
                and _is_entity_type(str(o))):
            _, local = split_iri(str(o))
            by_type[local] = by_type.get(local, 0) + 1
            subjects.add(s)
            ns_counter[split_iri(str(s))[0]] += 1

    base_iri = ns_counter.most_common(1)[0][0] if ns_counter else ""
    by_type = dict(sorted(by_type.items(), key=lambda kv: -kv[1]))

    # prefixes: read the turtle @prefix header directly (faithful to the dataset's
    # own naming — rdflib rebinds/injects ~25 defaults) and keep only the ones the
    # data actually uses
    declared: Dict[str, str] = {}  # namespace -> prefix
    header = graph_path.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r'@prefix\s+([A-Za-z][\w.\-]*)\s*:\s*<([^>]+)>\s*\.', header):
        declared[m.group(2)] = m.group(1)
    prefixes = {declared[ns]: ns for ns in used_ns if ns in declared}
    prefixes = dict(sorted(prefixes.items()))
    return base_iri, by_type, len(subjects), prefixes


# ── entities and chunk links from the graph ──────────────────────────

def _entity_records(g) -> List[dict]:
    """One entities.jsonl record per instance: IRI, types, label and the slugs of
    the sources attesting / defining it (orp:attestedIn / orp:definedIn)."""
    from rdflib import URIRef

    slug_of = {s: str(o) for s, o in g.subject_objects(URIRef(_DCTERMS + "identifier"))}
    records = []
    for s in sorted({s for s, o in g.subject_objects(URIRef(_RDF_TYPE))
                     if isinstance(s, URIRef) and _is_entity_type(str(o))}, key=str):
        types = sorted(t for t in {str(o) for o in g.objects(s, URIRef(_RDF_TYPE))}
                       if _is_entity_type(t))
        if not types:
            continue
        label = g.value(s, URIRef(_RDFS_LABEL))
        rec: Dict[str, Any] = {"iri": str(s), "types": types,
                               "label": str(label) if label is not None else split_iri(str(s))[1]}
        for key, pred in (("attestedIn", "attestedIn"), ("definedIn", "definedIn")):
            slugs = sorted({slug_of.get(o, split_iri(str(o))[1])
                            for o in g.objects(s, URIRef(_ORP + pred))})
            if slugs:
                rec[key] = slugs
        records.append(rec)
    return records


def _chunk_entities(g) -> Dict[str, List[str]]:
    """chunk id -> IRIs of the entities mentioned in it, via orp:Mention."""
    from rdflib import URIRef

    links: Dict[str, set] = defaultdict(set)
    ident = URIRef(_DCTERMS + "identifier")
    for m, chunk in g.subject_objects(URIRef(_ORP + "inChunk")):
        cid = g.value(chunk, ident)
        for e in g.objects(m, URIRef(_ORP + "mentions")):
            if cid is not None:
                links[str(cid)].add(str(e))
    return {k: sorted(v) for k, v in links.items()}


# ── chunks ───────────────────────────────────────────────────────────

def _read_jsonl(paths: Iterable[Path]) -> Iterable[dict]:
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _working_chunks(root: Path) -> List[dict]:
    """The pipeline's DTO chunks: content/chunks.jsonl, else content/dto/chunks/*."""
    single = root / "content" / "chunks.jsonl"
    if single.exists():
        return list(_read_jsonl([single]))
    return list(_read_jsonl(sorted(Path(p) for p in glob.glob(str(root / "content/dto/chunks/*.jsonl")))))


def _spec_chunk(dto: dict, slug: str, entities: List[str]) -> dict:
    prov = dto.get("provenance") or {}
    rec: Dict[str, Any] = {"id": dto["chunk_id"], "doc": slug,
                           "seq": int(dto.get("chunk_index") or 0)}
    if prov.get("section"):
        rec["heading_path"] = [str(prov["section"])]
    if isinstance(prov.get("page"), int) and prov["page"] >= 1:
        rec["page"] = prov["page"]
        if prov.get("page_label"):
            rec["page_label"] = str(prov["page_label"])
    rec["text"] = dto.get("text", "")
    rec["n_words"] = len(rec["text"].split())
    if entities:
        rec["entities"] = entities
    return rec


# ── the packager ─────────────────────────────────────────────────────

def complete_dataset(root: str | Path, graph_rel: str = "ontology/world.ttl",
                     name: Optional[str] = None, license: Optional[str] = None,
                     base_iri: Optional[str] = None, version: Optional[str] = None,
                     builder: Optional[str] = None) -> dict:
    """Write the published files of a dataset directory and return its manifest.

    Writes content/chunks/<pack>.jsonl, content/sources.json (unless one in another
    shape exists), content/books.json, ontology/entities.jsonl, ontology/prefixes.json
    (unless present) and manifest.json.
    """
    from rdflib import Graph

    root = Path(root).resolve()
    graph_path = root / graph_rel
    if not graph_path.exists():
        raise RuntimeError(
            f"graph not found at '{graph_rel}' — pass --graph to point at your world/instance TTL")

    manifest_path = root / "manifest.json"
    manifest: Dict[str, Any] = (json.loads(manifest_path.read_text(encoding="utf-8"))
                                if manifest_path.exists() else {})

    g = Graph()
    g.parse(str(graph_path), format="turtle")
    inferred_base, by_type, entity_count, prefixes = infer_graph_stats(graph_path)

    # entities
    entities = _entity_records(g)
    (root / ENTITY_INDEX).parent.mkdir(parents=True, exist_ok=True)
    with open(root / ENTITY_INDEX, "w", encoding="utf-8") as f:
        for rec in entities:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # chunks, one file per pack (= source document)
    links = _chunk_entities(g)
    by_pack: Dict[str, List[dict]] = defaultdict(list)
    docs: Dict[str, dict] = {}
    for dto in _working_chunks(root):
        prov = dto.get("provenance") or {}
        slug = doc_slug(prov.get("source_path"), dto.get("document_id", ""))
        by_pack[slug].append(_spec_chunk(dto, slug, links.get(dto["chunk_id"], [])))
        docs.setdefault(slug, {"title": doc_title(prov.get("source_path")),
                               "path": prov.get("source_path") or "",
                               "document_id": dto.get("document_id", "")})
    chunks_dir = root / CHUNKS_DIR
    if by_pack:
        chunks_dir.mkdir(parents=True, exist_ok=True)
        for old in chunks_dir.glob("*.jsonl"):
            if old.stem not in by_pack:
                old.unlink()
        for slug, recs in by_pack.items():
            recs.sort(key=lambda r: r["seq"])
            with open(chunks_dir / f"{slug}.jsonl", "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    chunk_count = sum(len(v) for v in by_pack.values()) or sum(
        1 for _ in _read_jsonl(sorted(chunks_dir.glob("*.jsonl"))))

    # source and pack registries
    sources_path = root / "content" / "sources.json"
    existing = (json.loads(sources_path.read_text(encoding="utf-8"))
                if sources_path.exists() else {})
    if isinstance(existing, dict) and docs:
        for slug, meta in docs.items():
            existing.setdefault(slug, {}).update({k: v for k, v in meta.items() if v})
            existing[slug]["chunks"] = len(by_pack[slug])
        sources_path.parent.mkdir(parents=True, exist_ok=True)
        sources_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    books_path = root / "content" / "books.json"
    books = json.loads(books_path.read_text(encoding="utf-8")) if books_path.exists() else {}
    for slug, meta in docs.items():
        books.setdefault(slug, {"title": meta["title"], "path": meta["path"], "requires": []})
    if books:
        books_path.write_text(json.dumps(books, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # prefixes
    if not (root / "ontology" / "prefixes.json").exists() and prefixes:
        (root / "ontology" / "prefixes.json").write_text(
            json.dumps(prefixes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # manifest: merge, never delete
    dataset = manifest.setdefault("dataset", {})
    dataset.setdefault("id", dataset.get("slug") or _slugify(name or root.name))
    dataset.setdefault("name", name or dataset.get("slug") or root.name)
    if name:
        dataset["name"] = name
    if version:
        dataset["version"] = version
    dataset.setdefault("version", "0.1.0")
    if license:
        dataset["license"] = license
    dataset["built_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    dataset["builder"] = builder or _builder()

    manifest["ontorag"] = SPEC_VERSION
    ontology = manifest.setdefault("ontology", {})
    ontology.update({
        "format": "text/turtle", "graph": graph_rel, "entity_index": ENTITY_INDEX,
        "base_iri": base_iri or ontology.get("base_iri") or dataset.get("base_iri") or inferred_base,
        "counts": {"entities": entity_count, "by_type": by_type},
    })
    if (root / "ontology" / "prefixes.json").exists():
        ontology["prefixes"] = "ontology/prefixes.json"
    if (root / "ontology" / "schema.ttl").exists():
        ontology["schema"] = "ontology/schema.ttl"

    content = manifest.setdefault("content", {})
    content.update({"format": "application/x-ndjson", "chunks_glob": f"{CHUNKS_DIR}/*.jsonl"})
    if sources_path.exists():
        content["sources"] = "content/sources.json"
    if books_path.exists():
        content["books"] = "content/books.json"
    content["counts"] = {"documents": len(books) or len(docs), "chunks": chunk_count}

    embeddings = _embeddings_block(root)
    if embeddings:
        manifest["embeddings"] = {**manifest.get("embeddings", {}), **embeddings}

    manifest["schema"] = {k: f"{SPEC_BASE}{k}.schema.json"
                          for k in ("manifest", "chunk", "embedding", "entity")}
    if books:
        manifest["composition"] = {
            "model": "packs", "version": SPEC_VERSION, "unit": "source document",
            "registry": "content/books.json", "dependency": "orp:requires",
            "core": sorted(s for s, b in books.items() if not b.get("requires")),
            "scope": ("Access scope: the selected packs plus the spine, never closed over "
                      "requires. Composition: the selected packs closed over requires. "
                      "See https://ontorag.org/provenance/#packs"),
            "counts": {"packs": len(books)},
        }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    _log.info("dataset complete: %d entities, %d chunks in %d pack(s)",
              len(entities), chunk_count, len(books))
    return manifest


def _embeddings_block(root: Path) -> Optional[dict]:
    """Describe vector shards if the dataset has them (embeddings/vectors/*.jsonl)."""
    shards = sorted((root / "embeddings" / "vectors").glob("*.jsonl"))
    if not shards:
        return None
    first = next(_read_jsonl(shards[:1]), None)
    if not first or not first.get("vector"):
        return None
    block: Dict[str, Any] = {"vectors_glob": "embeddings/vectors/*.jsonl",
                             "dim": len(first["vector"]), "metric": "cosine",
                             "alignment": "by_id"}
    config = root / "embeddings" / "config.json"
    if config.exists():
        block["config"] = "embeddings/config.json"
        cfg = json.loads(config.read_text(encoding="utf-8"))
        for key in ("provider", "model", "metric", "normalized"):
            if key in cfg:
                block[key] = cfg[key]
    return block


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "dataset"


def _builder() -> str:
    try:
        from importlib.metadata import version
        return f"ontorag {version('ontorag')}"
    except Exception:
        return "ontorag"
