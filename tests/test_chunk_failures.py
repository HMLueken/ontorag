"""What happens when the LLM returns nothing usable for a chunk.

Both extractors already *intended* to skip such a chunk — "a single unparseable
chunk shouldn't abort schema induction" is in the code, and `map_chunks` does
filter the Nones out of its return value. The hole was the *progress* path:
`map_chunks` reports every completion, a skipped chunk arrives at the callback as
`None`, and the CLI's progress line called `.get()` on it. So one unparseable
chunk raised `AttributeError` from inside the completion loop and killed a run
that had already paid for every other chunk — which is how a 25-document pilot
spent $1.07 and wrote nothing at all. On 5,000 abstracts that is hours of spend
discarded at the last step.

These tests pin the whole path: the callback survives a failed chunk on both the
sequential and threaded paths, the extractors name what they skipped, the
aggregator tolerates a store with `null` lines in it, and the pipeline ledger
leaves a document out when nothing came of it so the next run retries it.
"""
import json

import pytest

from ontorag import proposal_aggregator as agg


def test_aggregator_survives_a_poisoned_store():
    """`null` lines exist in raw stores written before failures were dropped."""
    out = agg.aggregate_chunk_proposals([
        {"chunk_id": "a", "proposed_additions": {"classes": [{"name": "Gene"}]}},
        None,
        {"chunk_id": "b", "proposed_additions": {"classes": [{"name": "Protein"}]}},
    ])
    names = {c["name"] for c in out["classes"]}
    assert names == {"Gene", "Protein"}


def test_progress_callback_survives_a_failed_chunk(monkeypatch):
    """The bug that cost a run: `map_chunks` reports every completion, including a
    chunk that returned nothing, and the CLI's progress line called `.get()` on it.

    Runs at concurrency 2 as well, because the threaded path is the one production
    uses and the sequential path is the one tests used."""
    from ontorag import ontology_extractor_openrouter as ose

    def fake_chat(system, user):
        if "POISON" in user:
            raise ValueError("model returned empty/null content")
        return {"proposed_additions": {"classes": [{"name": "Thing"}]}}

    monkeypatch.setattr(ose, "_chat_json", fake_chat)
    monkeypatch.setattr(ose.time, "sleep", lambda *_: None)

    seen = []

    def progress(idx, total, chunk_id, data):
        seen.append((chunk_id, len(data.get("proposed_additions", {}).get("classes", []))))

    chunks = [{"chunk_id": f"c{i}", "text": "POISON" if i == 1 else "fine"} for i in range(4)]
    for concurrency in (1, 2):
        seen.clear()
        out = ose.extract_schema_chunk_proposals(chunks, {}, on_chunk_done=progress,
                                                 concurrency=concurrency)
        assert len(out) == 3, f"concurrency={concurrency}"
        assert [c for c, _ in seen] == [c for c, _ in seen if c != "c1"], \
            "a chunk that produced nothing has nothing to report"
        assert ose.LAST_SKIPPED == ["c1"]


def test_schema_extractor_drops_and_reports_failures(monkeypatch):
    from ontorag import ontology_extractor_openrouter as ose

    calls = {"n": 0}

    def fake_chat(system, user):
        calls["n"] += 1
        if "POISON" in user:
            raise ValueError("model returned empty/null content")
        return {"proposed_additions": {"classes": [{"name": "Thing"}]}, "warnings": []}

    monkeypatch.setattr(ose, "_chat_json", fake_chat)
    monkeypatch.setattr(ose.time, "sleep", lambda *_: None)   # no retry backoff in tests

    chunks = [
        {"chunk_id": "good-1", "text": "fine"},
        {"chunk_id": "bad-1", "text": "POISON"},
        {"chunk_id": "good-2", "text": "fine"},
    ]
    out = ose.extract_schema_chunk_proposals(chunks, {"namespace": "http://x/"}, concurrency=1)

    assert len(out) == 2, "the failed chunk must not be returned"
    assert all(isinstance(r, dict) for r in out)
    assert ose.LAST_SKIPPED == ["bad-1"], "and the caller must be able to see which"


def test_instance_extractor_drops_and_reports_failures(monkeypatch):
    from ontorag import instance_extractor_openrouter as ise

    def fake_chat(system, user):
        if "POISON" in user:
            raise ValueError("unparseable")
        return {"instances": [{"class": "Gene", "label": "BRCA1"}]}

    monkeypatch.setattr(ise, "_chat_json", fake_chat)
    monkeypatch.setattr(ise.time, "sleep", lambda *_: None)

    chunks = [{"chunk_id": "ok", "text": "fine"}, {"chunk_id": "nope", "text": "POISON"}]
    out = ise.extract_instance_chunk_proposals(chunks, {"classes": [{"name": "Gene"}]},
                                               concurrency=1)
    assert len(out) == 1 and isinstance(out[0], dict)
    assert ise.LAST_SKIPPED == ["nope"]


def test_skip_list_is_cleared_by_a_clean_run(monkeypatch):
    """Otherwise a later run records documents as failed because an earlier one did."""
    from ontorag import ontology_extractor_openrouter as ose

    monkeypatch.setattr(ose, "_chat_json",
                        lambda s, u: {"proposed_additions": {"classes": []}})
    monkeypatch.setattr(ose.time, "sleep", lambda *_: None)
    ose.LAST_SKIPPED = ["stale"]
    ose.extract_schema_chunk_proposals([{"chunk_id": "c", "text": "x"}], {}, concurrency=1)
    assert ose.LAST_SKIPPED == []


def test_read_jsonl_skips_null_lines(tmp_path):
    from ontorag.cli import read_jsonl

    p = tmp_path / "raw.jsonl"
    p.write_text('{"chunk_id": "a"}\nnull\n{"chunk_id": "b"}\n', encoding="utf-8")
    assert [r["chunk_id"] for r in read_jsonl(str(p))] == ["a", "b"]


def test_failed_documents_stay_out_of_the_ledger(tmp_path):
    """A document nothing was extracted from must be retried, not recorded done."""
    from ontorag import run_stage

    root = tmp_path
    (root / "ontology").mkdir()
    (root / "ontology" / "skipped_chunks.json").write_text(
        json.dumps({"extract-schema": ["doc_b#c0"]}), encoding="utf-8")

    chunks = [
        {"chunk_id": "doc_a#c0", "document_id": "doc_a"},
        {"chunk_id": "doc_b#c0", "document_id": "doc_b"},
    ]
    failed = run_stage._skipped_docs(root, "extract-schema", chunks)
    assert failed == {"doc_b"}

    run_stage._record_done(root, "extract-schema",
                           [d["document_id"] for d in chunks
                            if d["document_id"] not in failed])
    state = json.loads((root / run_stage.STATE_FILE).read_text(encoding="utf-8"))
    assert state["extract-schema"]["documents"] == ["doc_a"]


def test_no_skip_file_means_nothing_skipped(tmp_path):
    from ontorag import run_stage

    assert run_stage._skipped_docs(tmp_path, "extract-schema", [{"chunk_id": "x"}]) == set()


def test_llm_names_become_valid_iris():
    """A model proposes what it reads. `BC1 RNA` in an IRI is not a cosmetic
    problem: rdflib refuses to serialise the graph, at write time, after every
    chunk has been paid for."""
    from rdflib import Graph

    from ontorag.instances_to_ttl import instance_proposals_to_graph

    ns = "http://ontorag.dev/beir/scifact/"
    proposals = [{
        "chunk_id": "c0",
        "instances": [
            {"class": "BC1 RNA", "label": "BC1 RNA", "properties": {"found in": "brain"},
             "relations": [{"predicate": "expressed in", "target_class": "Cell type",
                            "target_label": "neuron"}]},
            {"class": "α-synuclein", "label": "α-synuclein"},
        ],
    }]
    g = instance_proposals_to_graph({"c0": {"text": "…"}}, proposals, namespace=ns)
    ttl = g.serialize(format="turtle")          # the step that used to raise
    assert Graph().parse(data=ttl, format="turtle"), "and it must round-trip"


def test_schema_and_instances_agree_on_iris():
    """If the two modules sanitise differently, instances are typed with a class
    the schema never defines — no error, no results, nothing to see."""
    from ontorag.iri import local_name
    from ontorag.instances_to_ttl import instance_proposals_to_graph
    from ontorag.proposal_to_ttl import proposal_to_ttl

    ns = "http://x/"
    schema = proposal_to_ttl({"classes": [{"name": "BC1 RNA", "description": "d"}]}, biz_ns=ns)
    schema_iris = {str(s) for s in schema.subjects()}

    world = instance_proposals_to_graph(
        {"c0": {"text": "…"}},
        [{"chunk_id": "c0", "instances": [{"class": "BC1 RNA", "label": "BC1 RNA"}]}],
        namespace=ns)
    typed_as = {str(o) for o in world.objects(None, None) if str(o).startswith(ns)}

    expected = f"{ns}{local_name('BC1 RNA')}"
    assert expected in schema_iris, f"schema defines {schema_iris}"
    assert expected in typed_as, "instances must be typed with the class the schema defines"


def test_local_name_is_stable_and_safe():
    from ontorag.iri import local_name

    assert local_name("Character") == "Character"          # the common case is untouched
    assert local_name("BC1 RNA") == "BC1_RNA"
    assert local_name("T-cell receptor (TCR)") == "T-cell_receptor_%28TCR%29"
    assert local_name("  ") .startswith("_")                # unnamed still gets an identity
    assert local_name("A B") == local_name("A  B")          # whitespace runs collapse


def test_list_valued_property_fields_expand(monkeypatch):
    """Nothing binds a model to the schema it was asked for. Over one 5,000-abstract
    corpus, 5,175 `domain`/`name`/`range` fields came back as lists — the aggregator
    called `.strip()` on a list and the run died at the merge, fully paid for.

    A list of domains is a real claim about several domains, so it expands."""
    out = agg.aggregate_chunk_proposals([{
        "chunk_id": "c0",
        "proposed_additions": {
            "object_properties": [
                {"name": "treats", "domain": ["Drug", "Therapy"], "range": "Disease"},
            ],
            "datatype_properties": [
                {"name": ["dosage"], "domain": "Drug", "range": "string"},
            ],
        },
    }])
    pairs = {(p["domain"], p["name"], p["range"]) for p in out["object_properties"]}
    assert pairs == {("Drug", "treats", "Disease"), ("Therapy", "treats", "Disease")}
    assert [(p["domain"], p["name"]) for p in out["datatype_properties"]] == [("Drug", "dosage")]


def test_expansion_is_capped():
    """A model having a bad day should not turn one proposal into a thousand."""
    many = [f"C{i}" for i in range(50)]
    out = agg.aggregate_chunk_proposals([{
        "chunk_id": "c0",
        "proposed_additions": {"object_properties": [
            {"name": "rel", "domain": many, "range": many}]},
    }])
    assert len(out["object_properties"]) <= 25


def test_unusable_property_shapes_are_skipped_not_fatal():
    out = agg.aggregate_chunk_proposals([{
        "chunk_id": "c0",
        "proposed_additions": {"object_properties": [
            {"name": "rel", "domain": None, "range": "X"},
            {"name": "", "domain": "A", "range": "B"},
            {"name": "ok", "domain": "A", "range": "B"},
        ]},
    }])
    assert [p["name"] for p in out["object_properties"]] == ["ok"]
