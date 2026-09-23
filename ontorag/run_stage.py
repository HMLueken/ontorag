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
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

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
    try:
        subprocess.run(cmd, check=True, cwd=root)
    except subprocess.CalledProcessError as e:
        # A sub-step failing is an ordinary outcome -- an unreachable endpoint, a
        # bad key, a model the server does not serve. Surfacing it as a traceback
        # is useless in an Actions log and worse in a desktop UI, so report which
        # step failed and stop.
        step = next((a for a in args if not a.startswith("-")), "step")
        raise SystemExit(
            f"stage failed at `{step}` (exit {e.returncode}). "
            f"Check the LLM endpoint and key for this dataset, then re-run."
        ) from None


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


# ── incremental extraction ───────────────────────────────────────────
#
# Ingest is already content-addressed: re-ingesting an unchanged file is a no-op.
# The expensive half was not — adding one document re-ran the LLM over every chunk
# in the corpus. The ledger below records which *documents* each LLM stage has
# already seen, so a re-run only pays for what is new.
#
# Documents, not chunks, because a document is what a user adds and because
# `stable_document_id` is a content hash: edit a file and it becomes a new
# document, which is exactly when its chunks must be re-extracted.
#
# KNOWN LIMITATION: an *edited* file is re-extracted, but what its previous
# version contributed is not removed. `world.ttl` is merged, so the graph then
# holds instances from both revisions. Chunk ids are derived from the source slug
# rather than the document id, so there is no reliable key to delete the
# superseded triples by; until there is, `--full` is the remedy after edits.
# Purely *adding* files -- the common case, and the one this is for -- is exact.

STATE_FILE = "ontology/extraction_state.json"


def _load_state(root: pathlib.Path) -> dict:
    p = root / STATE_FILE
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _log.warning("%s is unreadable; treating every document as new", STATE_FILE)
        return {}


def _record_done(root: pathlib.Path, step: str, doc_ids: List[str]) -> None:
    state = _load_state(root)
    entry = state.setdefault(step, {"documents": []})
    seen = set(entry.get("documents", []))
    seen.update(doc_ids)
    entry["documents"] = sorted(seen)
    entry["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p = root / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _read_chunks(path: pathlib.Path) -> List[dict]:
    if not path.is_file():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def pending_chunks(root: pathlib.Path, step: str, full: bool = False):
    """Chunks this step has not processed yet, and the documents they belong to.

    Returns (chunks, doc_ids). With `full`, everything is pending — the escape
    hatch for a changed schema card or a corrected prompt, where past extractions
    are no longer comparable.
    """
    chunks = _read_chunks(root / "content" / "chunks.jsonl")
    if full:
        return chunks, sorted({c.get("document_id", "") for c in chunks if c.get("document_id")})
    done = set(_load_state(root).get(step, {}).get("documents", []))
    fresh = [c for c in chunks if c.get("document_id") and c["document_id"] not in done]
    return fresh, sorted({c["document_id"] for c in fresh})


def _card_hash(root: pathlib.Path, card_rel: str) -> str:
    p = root / card_rel
    if not p.is_file():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _card_changed(root: pathlib.Path, card_rel: str) -> bool:
    state = _load_state(root)
    recorded = state.get("extract-instances", {}).get("card")
    current = _card_hash(root, card_rel)
    return bool(recorded) and recorded != current


def _write_pending(root: pathlib.Path, chunks: List[dict], name: str) -> str:
    """Materialise the pending chunks so the CLI command can be handed a file."""
    rel = f"content/{name}"
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return rel


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


def propose(root: pathlib.Path, m: dict, full: bool = False) -> None:
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

    pending, doc_ids = pending_chunks(root, "extract-schema", full=full)
    if not pending:
        print("extract-schema: nothing new to induce (all documents already seen)",
              flush=True)
    else:
        chunks_arg = _write_pending(root, pending, "chunks.pending.jsonl")
        print(f"extract-schema: {len(pending)} chunk(s) from {len(doc_ids)} new "
              f"document(s)", flush=True)
        _engine(root, "extract-schema", "--chunks", chunks_arg,
                "--schema-card", "ontology/schema_card.baseline.json",
                "--raw-in", "ontology/proposals.raw.jsonl",
                "--raw-out", "ontology/proposals.raw.jsonl",
                "--out", "ontology/proposal.json", llm=True, m=m)
        _record_done(root, "extract-schema", doc_ids)
    _engine(root, "align-schema", "--proposal", "ontology/proposal.json",
            "--baseline", "ontology/schema_card.baseline.json",
            "--out", "ontology/alignment.json", llm=True, m=m)
    _engine(root, "build-schema-card", "--previous", "ontology/schema_card.baseline.json",
            "--proposal", "ontology/alignment.json",
            "--original-proposal", "ontology/proposal.json",
            "--out", "ontology/schema_card.proposed.json", m=m)
    save_state(root, m, "proposed")


def extract(root: pathlib.Path, m: dict, full: bool = False) -> None:
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
    # Instances are extracted *against* the card, so approving a different model
    # makes previous extractions incomparable -- fall back to a full pass rather
    # than leaving a graph that is half one schema and half another.
    if not full and _card_changed(root, card):
        print("schema card changed since the last extraction — running a full pass",
              flush=True)
        full = True
    pending, doc_ids = pending_chunks(root, "extract-instances", full=full)
    if not pending:
        print("extract-instances: nothing new to extract (all documents already seen)",
              flush=True)
    else:
        chunks_arg = _write_pending(root, pending, "chunks.pending.jsonl")
        print(f"extract-instances: {len(pending)} chunk(s) from {len(doc_ids)} new "
              f"document(s)", flush=True)
        args = ["extract-instances", "--chunks", chunks_arg,
                "--schema-card", card, "--out-ttl", "ontology/world.ttl"]
        if not full:
            args.append("--merge")   # union with what is already in the graph
        _engine(root, *args, llm=True, m=m)
        _record_done(root, "extract-instances", doc_ids)
        st = _load_state(root)
        st.setdefault("extract-instances", {})["card"] = _card_hash(root, card)
        (root / STATE_FILE).write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
    if (root / "ontology" / "world.ttl").exists():
        # derive the published files (dataset format 0.1) so the result can be served
        from ontorag.dataset_package import complete_dataset
        complete_dataset(root)
        m = load_manifest(root)
    save_state(root, m, "extracted")


STAGES = {"propose": propose, "extract": extract}


def run(stage: str, root: Optional[str] = None, full: bool = False) -> None:
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    path = pathlib.Path(root or ".").resolve()
    if not (path / "manifest.json").is_file():
        raise SystemExit(f"no manifest.json in {path} — is this a dataset directory?")
    m = load_manifest(path)
    _log.info("running stage %s in %s", stage, path)
    STAGES[stage](path, m, full=full)
    print(f"stage {stage} complete", flush=True)
