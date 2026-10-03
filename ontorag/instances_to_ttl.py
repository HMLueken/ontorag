from __future__ import annotations
import hashlib
from typing import Dict, Any, List, Optional

from urllib.parse import quote as _quote

from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import DCTERMS, RDF, RDFS, XSD

from ontorag.iri import local_name
from ontorag.sources import doc_slug, doc_title, file_sha256
from ontorag.verbosity import get_logger

_log = get_logger("ontorag.instances_to_ttl")

PROV = Namespace("http://www.w3.org/ns/prov#")
# Provenance follows the OntoRAG Provenance and Citation Ontology
# (https://ontorag.org/provenance/): a mention is a Web Annotation whose target
# locates the quoted passage in a source file.
ORP = Namespace("https://ontorag.org/provenance#")
OA = Namespace("http://www.w3.org/ns/oa#")

def _slug(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum() or ch in ("_","-")).strip("_-")

def _stable_instance_iri(ns: str, class_name: str, label: str, chunk_id: str) -> str:
    # the hash is over the *raw* names, so identity does not change with the
    # sanitiser; only the readable part of the path is made IRI-safe
    base = f"{class_name}|{label}|{chunk_id}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{ns}{local_name(class_name)}/{h}"

def instance_proposals_to_graph(
    chunk_dtos_by_id: Dict[str, Dict[str, Any]],
    proposals: List[Dict[str, Any]],
    namespace: str,
) -> Graph:
    _log.info("Converting %d proposals to RDF (namespace=%s)", len(proposals), namespace)

    BIZ = Namespace(namespace)
    g = Graph()
    g.bind("biz", BIZ)
    g.bind("prov", PROV)
    g.bind("rdfs", RDFS)
    g.bind("orp", ORP)
    g.bind("oa", OA)
    g.bind("dcterms", DCTERMS)
    described: set = set()      # sources/files/packs/chunks already in the graph

    instance_count = 0
    for cp in proposals:
        chunk_id = cp.get("chunk_id", "")
        chunk = chunk_dtos_by_id.get(chunk_id, {})
        prov = (chunk.get("provenance") or {})

        for inst in (cp.get("instances") or []):
            cls_name = inst.get("class", "").strip()
            if not cls_name:
                continue

            label = (inst.get("label") or inst.get("id_hint") or "").strip()
            iri = _stable_instance_iri(namespace, cls_name, label, chunk_id)
            s = URIRef(iri)

            g.add((s, RDF.type, URIRef(f"{namespace}{local_name(cls_name)}")))
            if label:
                g.add((s, RDFS.label, Literal(label)))
            instance_count += 1
            _log.debug("  instance: %s (%s) label=%r", cls_name, iri, label)

            # attributes (datatype properties): stored as literals
            attrs = inst.get("attributes", {}) or {}
            for prop_name, value in attrs.items():
                prop_name = (prop_name or "").strip()
                if not prop_name or value is None or value == "":
                    continue
                p = URIRef(f"{namespace}{local_name(prop_name)}")
                g.add((s, p, Literal(str(value))))

            # relations (object properties): create target nodes (lightweight) if needed
            for rel in inst.get("relations", []) or []:
                pred = (rel.get("predicate") or "").strip()
                tgt_cls = (rel.get("target_class") or "").strip()
                if not pred or not tgt_cls:
                    continue

                tgt_label = (rel.get("target_label") or rel.get("target_id_hint") or "").strip()
                tgt_iri = _stable_instance_iri(namespace, tgt_cls, tgt_label, chunk_id)
                t = URIRef(tgt_iri)

                g.add((t, RDF.type, URIRef(f"{namespace}{local_name(tgt_cls)}")))
                if tgt_label:
                    g.add((t, RDFS.label, Literal(tgt_label)))

                g.add((s, URIRef(f"{namespace}{local_name(pred)}"), t))

            # provenance: one orp:Mention per quoted passage
            mentions = [q for q in ((m.get("quote") or "").strip()
                                    for m in (inst.get("mentions") or [])) if q]
            if mentions:
                source, file_, chunk_uri = _describe_origin(g, namespace, chunk, chunk_id, described)
                g.add((s, ORP.attestedIn, source))
                for quote in mentions:
                    _add_mention(g, namespace, s, quote, file_, chunk_uri, prov)

    _log.info("Instance graph built: %d instances, %d triples", instance_count, len(g))
    return g


def _describe_origin(g: Graph, ns: str, chunk: Dict[str, Any], chunk_id: str,
                     described: set):
    """Add (once) the source, file, pack and chunk a passage comes from, and return
    (source, file, chunk) IRIs. One pack per source document, as in the dataset
    format, so a dataset can be scoped per document."""
    prov = chunk.get("provenance") or {}
    source_path = prov.get("source_path")
    document_id = chunk.get("document_id") or chunk_id.split("#", 1)[0]
    slug = doc_slug(source_path, document_id)

    source = URIRef(f"{ns}source/{slug}")
    file_ = URIRef(f"{ns}file/{document_id}")
    pack = URIRef(f"{ns}pack/{slug}")
    chunk_node = URIRef(f"{ns}chunk/{_quote(chunk_id, safe='')}")

    if slug not in described:
        described.add(slug)
        g.add((source, RDF.type, ORP.Source))
        g.add((source, DCTERMS.title, Literal(doc_title(source_path))))
        g.add((source, DCTERMS.identifier, Literal(slug)))
        g.add((file_, RDF.type, ORP.SourceFile))
        g.add((file_, ORP.fileOf, source))
        digest = file_sha256(source_path)
        if digest:
            g.add((file_, ORP.checksum, Literal(f"sha256:{digest}")))
        if prov.get("source_mime"):
            g.add((file_, DCTERMS["format"], Literal(prov["source_mime"])))
        g.add((pack, RDF.type, ORP.Pack))
        g.add((pack, ORP.packOf, source))
        g.add((pack, ORP.namedGraph, URIRef(f"{ns}graph/{slug}")))

    if chunk_id not in described:
        described.add(chunk_id)
        g.add((chunk_node, RDF.type, ORP.Chunk))
        g.add((chunk_node, DCTERMS.identifier, Literal(chunk_id)))
        g.add((chunk_node, ORP.inPack, pack))
        g.add((chunk_node, OA.hasSource, file_))
    return source, file_, chunk_node


def _add_mention(g: Graph, ns: str, instance: URIRef, quote: str, file_: URIRef,
                 chunk: URIRef, prov: Dict[str, Any]) -> None:
    key = hashlib.sha1(f"{instance}|{chunk}|{quote}".encode("utf-8")).hexdigest()[:16]
    mention = URIRef(f"{ns}mention/{key}")
    g.add((mention, RDF.type, ORP.Mention))
    g.add((mention, ORP.mentions, instance))
    g.add((mention, OA.motivatedBy, OA.identifying))
    g.add((mention, ORP.inChunk, chunk))
    g.add((instance, ORP.hasMention, mention))

    target = BNode()
    g.add((mention, OA.hasTarget, target))
    g.add((target, RDF.type, OA.SpecificResource))
    g.add((target, OA.hasSource, file_))

    sel = BNode()
    g.add((target, OA.hasSelector, sel))
    g.add((sel, RDF.type, OA.TextQuoteSelector))
    g.add((sel, OA.exact, Literal(quote)))

    page = prov.get("page")
    if isinstance(page, int) and page >= 1:
        sel = BNode()
        g.add((target, OA.hasSelector, sel))
        g.add((sel, RDF.type, ORP.PageSelector))
        g.add((sel, ORP.pageStart, Literal(page, datatype=XSD.integer)))
        if prov.get("page_label"):
            g.add((sel, ORP.pageLabel, Literal(str(prov["page_label"]))))

    if prov.get("section"):
        sel = BNode()
        g.add((target, OA.hasSelector, sel))
        g.add((sel, RDF.type, ORP.SectionSelector))
        g.add((sel, ORP.sectionTitle, Literal(str(prov["section"]))))
