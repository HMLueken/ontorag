import json
from pathlib import Path
from typing import Any, Dict
from ontorag import llm_config
from ontorag.jsonparse import loads_lenient
from ontorag.verbosity import get_logger
from requests_cache import NEVER_EXPIRE, CachedSession

_log = get_logger("ontorag.schema_alignment")

def _chat_json(system: str, user: str) -> Dict[str, Any]:
    key = llm_config.api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (set it in the environment or pass --api-key)")

    url = llm_config.base_url()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": llm_config.site_url(),
        "X-Title": llm_config.app_name(),
    }
    payload = {
        "thread": None,
        "model": llm_config.model(),
        "prompt" :user,
        "customInstructions": system
    }

    _log.debug("API request: model=%s prompt_len=%d", llm_config.model(), len(user))
    s = CachedSession(
        Path("responses.sqlite"),
        expire_after=NEVER_EXPIRE,
        allowable_methods=("GET", "POST")
    )
    r = s.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    chunks = r.text.split("\n")
    resp_json = json.loads(chunks[-2])
    content = str(resp_json.get("response", ""))
    if not content:  # some models (e.g. reasoning ones) can return null content
        raise RuntimeError("model returned empty/null content")
    _log.debug("API response: %d chars", len(content))
    # tolerant parse: recovers fenced / prose-wrapped / trailing-junk payloads
    return loads_lenient(content)