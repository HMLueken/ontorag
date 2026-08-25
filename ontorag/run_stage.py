"""Run one governed pipeline stage over a dataset directory.

This is the stage *composition* — which CLI commands make up `propose` and
`extract`, and in what order. It used to live as `.hub/run_stage.py` inside every
dataset repo, copied there by the Hub at provision time, which meant three
problems: the definition drifted between repos, editing it never reached repos
already provisioned, and running a stage required a Python interpreter separate
from the engine.

Keeping it in the engine fixes all three. The Actions workflow, the Hub and the
desktop client now call one definition, versioned with the code it drives — and a
frozen single-file build can run a stage with no Python installed at all, because
it re-invokes *itself* rather than a script.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import List, Optional

from ontorag.verbosity import get_logger

_log = get_logger("ontorag.run_stage")


def _self_cmd() -> List[str]:
    """How to re-invoke the engine.

    Frozen (PyInstaller), `sys.executable` *is* the ontorag binary, so it can call
    itself; that is what lets a packaged desktop app run stages without Python on
    the machine. Otherwise use the installed console script.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [shutil.which("ontorag") or "ontorag"]


def load_manifest(root: pathlib.Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def save_state(root: pathlib.Path, m: dict, state: str) -> None:
    m.setdefault("hub", {})["state"] = state
    m["hub"]["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root / "manifest.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _safe_prefix(slug: str) -> str:
    p = "".join(c if c.isalnum() else "_" for c in slug).strip("_")
    return p or "ds"


def _run(root: pathlib.Path, *args: str) -> None:
    cmd = [*_self_cmd(), *args]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=root)


def _engine(root: pathlib.Path, *args: str, llm: bool = False, m: dict) -> None:
    """Invoke a CLI command. LLM stages get model/base-url/concurrency flags from
    the manifest, so a dataset pointed at a local ollama stays pointed at it."""
    pre: List[str] = []
    if llm:
        ds = m["dataset"]
        if ds.get("openrouter_model"):
            pre += ["--model", ds["openrouter_model"]]
        if ds.get("openrouter_base_url"):
            pre += ["--base-url", ds["openrouter_base_url"]]
        pre += ["--concurrency", str(ds.get("concurrency", 4))]
    _run(root, *pre, *args)


def _ingest_and_chunk(root: pathlib.Path, m: dict) -> None:
    """Ingest every file in content/sources/ (content-hashed → new files only) and
    (re)build the single content/chunks.jsonl. Shared by propose + extract so the
    corpus can be extended over time and re-run."""
    sources = sorted(glob.glob(str(root / "content" / "sources" / "*")))
    if not sources:
        raise SystemExit("no files in content/sources/")
    for f in sources:
        _engine(root, "ingest", os.path.relpath(f, root), "--engine", "builtin",
                "--out", "content/dto", m=m)
    chunk_files = sorted(glob.glob(str(root / "content" / "dto" / "chunks" / "*.jsonl")))
    with open(root / "content" / "chunks.jsonl", "w", encoding="utf-8") as out:
        for cf in chunk_files:
            out.write(pathlib.Path(cf).read_text(encoding="utf-8"))


def propose(root: pathlib.Path, m: dict) -> None:
    ns = m["dataset"]["base_iri"]
    baselines = m.get("baselines", [])
    (root / "ontology" / "catalog").mkdir(parents=True, exist_ok=True)

    if baselines:
        for b in baselines:
            ttl = root / "ontology" / "baselines" / f"{b}.ttl"
            if ttl.exists():
                _engine(root, "register-ontology", b, f"ontology/baselines/{b}.ttl",
                        "--catalog", "ontology/catalog", m=m)
        _engine(root, "init-schema-card", "--baselines", ",".join(baselines),
                "--catalog", "ontology/catalog", "--namespace", ns,
                "--out", "ontology/schema_card.baseline.json", m=m)
    else:
        (root / "ontology" / "schema_card.baseline.json").write_text(
            json.dumps({"namespace": ns, "classes": [], "datatype_properties": [],
                        "object_properties": []}, indent=2), encoding="utf-8")

    _ingest_and_chunk(root, m)

    _engine(root, "extract-schema", "--chunks", "content/chunks.jsonl",
            "--schema-card", "ontology/schema_card.baseline.json",
            "--out", "ontology/proposal.json", llm=True, m=m)
    _engine(root, "align-schema", "--proposal", "ontology/proposal.json",
            "--baseline", "ontology/schema_card.baseline.json",
            "--out", "ontology/alignment.json", llm=True, m=m)
    _engine(root, "build-schema-card", "--previous", "ontology/schema_card.baseline.json",
            "--proposal", "ontology/alignment.json",
            "--original-proposal", "ontology/proposal.json",
            "--out", "ontology/schema_card.proposed.json", m=m)
    save_state(root, m, "proposed")


def extract(root: pathlib.Path, m: dict) -> None:
    ns = m["dataset"]["base_iri"]
    prefix = _safe_prefix(m["dataset"]["slug"])
    _ingest_and_chunk(root, m)  # pick up any files added since propose
    _engine(root, "export-schema-ttl", "--proposal", "ontology/alignment.json",
            "--original-proposal", "ontology/proposal.json",
            "--namespace", ns, "--prefix", prefix,
            "--catalog", "ontology/catalog",
            "--baseline", "ontology/schema_card.baseline.json",
            "--out", "ontology/schema.ttl", m=m)
    # the approved card if the human validated it, else the proposed one
    card = ("ontology/schema_card.json"
            if (root / "ontology" / "schema_card.json").exists()
            else "ontology/schema_card.proposed.json")
    _engine(root, "extract-instances", "--chunks", "content/chunks.jsonl",
            "--schema-card", card, "--out-ttl", "ontology/world.ttl", llm=True, m=m)
    save_state(root, m, "extracted")


STAGES = {"propose": propose, "extract": extract}


def run(stage: str, root: Optional[str] = None) -> None:
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    path = pathlib.Path(root or ".").resolve()
    if not (path / "manifest.json").is_file():
        raise SystemExit(f"no manifest.json in {path} — is this a dataset directory?")
    m = load_manifest(path)
    _log.info("running stage %s in %s", stage, path)
    STAGES[stage](path, m)
    print(f"stage {stage} complete", flush=True)
