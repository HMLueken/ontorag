"""Stage composition lives in the engine now — pin the command sequence.

Mirrors the old tests/test_runner.py in the Hub repo, which tested the copy that
used to live in each dataset repo. Same assertions, one definition.
"""
import json
from pathlib import Path

import pytest

from ontorag import run_stage


def _chunks(tmp_path, *docs):
    """Write the chunk artifacts a real `ontorag ingest` would leave behind.

    Both the per-document DTO files *and* the assembled chunks.jsonl, because
    _ingest_and_chunk rebuilds the latter from the former — writing only
    chunks.jsonl would see it truncated the moment a stage runs.
    """
    dto = tmp_path / "content" / "dto" / "chunks"
    dto.mkdir(parents=True, exist_ok=True)
    lines = []
    for doc in docs:
        rows = [json.dumps({"document_id": doc, "chunk_id": f"{doc}::{i}",
                            "text": "x", "chunk_index": i}) for i in range(2)]
        (dto / f"{doc}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        lines += rows
    (tmp_path / "content" / "chunks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup(tmp_path: Path) -> dict:
    (tmp_path / "content" / "sources").mkdir(parents=True)
    (tmp_path / "content" / "sources" / "a.md").write_text("# A\nsome text", encoding="utf-8")
    (tmp_path / "ontology" / "baselines").mkdir(parents=True)
    (tmp_path / "ontology" / "baselines" / "rpg.ttl").write_text("@prefix x: <urn:x#> .", encoding="utf-8")
    manifest = {
        "ontorag": "0.1", "hub": {"state": "corpus"},
        "dataset": {"slug": "my-ds", "title": "t", "base_iri": "https://ontorag.dev/my-ds/",
                    "openrouter_model": "m/x", "concurrency": 3},
        "baselines": ["rpg"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # _run is stubbed in these tests, so the real ingest never runs; write the
    # chunks it would have produced, otherwise the incremental check correctly
    # concludes there is nothing to extract and skips the LLM steps under test.
    _chunks(tmp_path, "doc_a")
    return manifest


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(run_stage, "_run", lambda root, *a: seen.append(" ".join(a)))
    return seen


def test_propose_assembly(tmp_path, calls):
    m = _setup(tmp_path)
    run_stage.propose(tmp_path, m)

    assert any("register-ontology rpg ontology/baselines/rpg.ttl" in c for c in calls)
    assert any("init-schema-card --baselines rpg" in c for c in calls)
    # one call for the directory, not one per file: a 5,000-document corpus paid
    # ~0.5s of interpreter start per document otherwise
    assert any(c.startswith("ingest content/sources ") or c == "ingest content/sources"
               for c in calls)
    assert not any("ingest content/sources/a.md" in c for c in calls)
    es = next(c for c in calls if "extract-schema" in c)
    # induction runs over the pending set, not the whole corpus
    assert "--chunks content/chunks.pending.jsonl" in es
    assert "--raw-in ontology/proposals.raw.jsonl" in es
    assert "--model m/x" in es and "--concurrency 3" in es   # LLM flags applied
    assert any("align-schema" in c for c in calls)
    assert any("build-schema-card" in c for c in calls)
    assert json.loads((tmp_path / "manifest.json").read_text())["hub"]["state"] == "proposed"


def test_extract_assembly(tmp_path, calls):
    m = _setup(tmp_path)
    run_stage.extract(tmp_path, m)

    exp = next(c for c in calls if "export-schema-ttl" in c)
    assert "--prefix my_ds" in exp          # slug sanitized for a TTL prefix
    assert "--namespace https://ontorag.dev/my-ds/" in exp
    ei = next(c for c in calls if "extract-instances" in c)
    assert "ontology/schema_card.proposed.json" in ei   # falls back to proposed card
    assert "--model m/x" in ei
    assert json.loads((tmp_path / "manifest.json").read_text())["hub"]["state"] == "extracted"


def test_unknown_stage_and_missing_manifest(tmp_path):
    with pytest.raises(SystemExit, match="unknown stage"):
        run_stage.run("drop-tables", str(tmp_path))
    with pytest.raises(SystemExit, match="no manifest.json"):
        run_stage.run("propose", str(tmp_path))


def test_frozen_build_reinvokes_itself(monkeypatch):
    """The property that lets a packaged desktop app run stages with no Python:
    a frozen binary calls sys.executable, not a console script."""
    monkeypatch.setattr(run_stage.sys, "frozen", True, raising=False)
    monkeypatch.setattr(run_stage.sys, "executable", "/Apps/OntoRAG.app/ontorag", raising=False)
    assert run_stage._self_cmd() == ["/Apps/OntoRAG.app/ontorag"]


def test_failing_step_reports_which_one(tmp_path, monkeypatch):
    """A dead endpoint or bad key is ordinary; it must not surface as a traceback."""
    import subprocess as sp
    m = _setup(tmp_path)

    def boom(cmd, **kw):
        raise sp.CalledProcessError(1, cmd)

    monkeypatch.setattr(run_stage.subprocess, "run", boom)
    monkeypatch.setattr(run_stage, "_self_cmd", lambda: ["ontorag"])
    with pytest.raises(SystemExit) as e:
        run_stage.propose(tmp_path, m)
    msg = str(e.value)
    assert "stage failed at" in msg and "exit 1" in msg, msg
    assert "Traceback" not in msg


# ── incremental extraction ───────────────────────────────────────────

def test_pending_is_everything_on_a_first_run(tmp_path):
    _chunks(tmp_path, "doc_a", "doc_b")
    chunks, docs = run_stage.pending_chunks(tmp_path, "extract-schema")
    assert docs == ["doc_a", "doc_b"] and len(chunks) == 4


def test_pending_excludes_documents_already_processed(tmp_path):
    _chunks(tmp_path, "doc_a", "doc_b")
    run_stage._record_done(tmp_path, "extract-schema", ["doc_a"])
    chunks, docs = run_stage.pending_chunks(tmp_path, "extract-schema")
    assert docs == ["doc_b"], "only the new document should be re-sent to the LLM"
    assert {c["document_id"] for c in chunks} == {"doc_b"}


def test_ledger_is_per_step(tmp_path):
    """Inducing the schema from a document does not mean instances were extracted."""
    _chunks(tmp_path, "doc_a")
    run_stage._record_done(tmp_path, "extract-schema", ["doc_a"])
    _, schema_docs = run_stage.pending_chunks(tmp_path, "extract-schema")
    _, inst_docs = run_stage.pending_chunks(tmp_path, "extract-instances")
    assert schema_docs == [] and inst_docs == ["doc_a"]


def test_full_ignores_the_ledger(tmp_path):
    _chunks(tmp_path, "doc_a", "doc_b")
    run_stage._record_done(tmp_path, "extract-schema", ["doc_a", "doc_b"])
    _, docs = run_stage.pending_chunks(tmp_path, "extract-schema", full=True)
    assert docs == ["doc_a", "doc_b"]


def test_unreadable_ledger_falls_back_to_reprocessing(tmp_path):
    """Corrupt state must cost money, not correctness."""
    _chunks(tmp_path, "doc_a")
    p = tmp_path / run_stage.STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json", encoding="utf-8")
    _, docs = run_stage.pending_chunks(tmp_path, "extract-schema")
    assert docs == ["doc_a"]


def test_propose_skips_the_llm_when_nothing_is_new(tmp_path, calls, capsys):
    m = _setup(tmp_path)
    run_stage.propose(tmp_path, m)          # first pass populates the ledger
    calls.clear()
    # the ingest step re-runs (cheap, no LLM) but induction must not
    run_stage.propose(tmp_path, m)
    assert not any("extract-schema" in c for c in calls), calls
    assert "nothing new to induce" in capsys.readouterr().out


def test_a_changed_schema_card_forces_a_full_instance_pass(tmp_path):
    _chunks(tmp_path, "doc_a")
    card = "ontology/schema_card.json"
    (tmp_path / "ontology").mkdir(parents=True, exist_ok=True)
    (tmp_path / card).write_text('{"classes": []}', encoding="utf-8")
    run_stage._record_done(tmp_path, "extract-instances", ["doc_a"])
    st = run_stage._load_state(tmp_path)
    st["extract-instances"]["card"] = run_stage._card_hash(tmp_path, card)
    (tmp_path / run_stage.STATE_FILE).write_text(json.dumps(st), encoding="utf-8")

    assert not run_stage._card_changed(tmp_path, card)
    (tmp_path / card).write_text('{"classes": [{"name": "Magus"}]}', encoding="utf-8")
    assert run_stage._card_changed(tmp_path, card), \
        "a graph half-extracted against one model and half another is worse than a re-run"


def test_no_baselines_means_no_alignment_and_a_real_card(tmp_path, calls):
    """A dataset without baselines used to end up with an empty schema card: the
    card was built from an alignment that had nothing to align against."""
    m = _setup(tmp_path)
    m["baselines"] = []
    (tmp_path / "manifest.json").write_text(json.dumps(m), encoding="utf-8")

    run_stage.propose(tmp_path, m)

    assert not any("align-schema" in c for c in calls), "nothing to align against"
    card = next(c for c in calls if "build-schema-card" in c)
    assert "--proposal ontology/proposal.json" in card, card
    assert "--proposal ontology/alignment.json" not in card

    calls.clear()
    run_stage.extract(tmp_path, m)
    ttl = next(c for c in calls if "export-schema-ttl" in c)
    assert "--proposal ontology/proposal.json" in ttl, ttl


def test_baselines_still_go_through_alignment(tmp_path, calls):
    m = _setup(tmp_path)
    run_stage.propose(tmp_path, m)
    assert any("align-schema" in c for c in calls)
    card = next(c for c in calls if "build-schema-card" in c)
    assert "--proposal ontology/alignment.json" in card, card
