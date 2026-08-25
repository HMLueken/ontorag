"""Stage composition lives in the engine now — pin the command sequence.

Mirrors the old tests/test_runner.py in the Hub repo, which tested the copy that
used to live in each dataset repo. Same assertions, one definition.
"""
import json
from pathlib import Path

import pytest

from ontorag import run_stage


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
    assert any(c.startswith("ingest content/sources/a.md") for c in calls)
    es = next(c for c in calls if "extract-schema" in c)
    assert "--chunks content/chunks.jsonl" in es
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
