import json
from pathlib import Path
from typing import Any, Dict
import unicodedata
from ontorag import llm_config
from ontorag.jsonparse import loads_lenient
from ontorag.verbosity import get_logger
from openai import OpenAI

_log = get_logger("ontorag.schema_alignment")

def _chat_json(system: str, user: str) -> Dict[str, Any]:
    Path("tmp").mkdir(exist_ok=True)
    path = Path("tmp/llm_response_cache.json")
    if path.exists():
        cache = json.loads(path.read_text())
    else:
        cache = dict()
    url = llm_config.base_url()
    cache_key = f"{system}\n{user}"
    if cache_key in cache:
        resp = cache[cache_key]
    else:
        _log.debug("API request: model=%s prompt_len=%d", llm_config.model(), len(user))
        with OpenAI(
            base_url=url,
            api_key=llm_config.api_key()
        ) as client:
            response = client.responses.create(
                model=llm_config.model(),
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            resp = unicodedata.normalize("NFKD", response.output_text)
        if path.exists():
            cache = json.loads(path.read_text())
        else:
            cache = dict()
        cache[cache_key] = resp
        path.write_text(json.dumps(cache))
    if not resp:  # some models (e.g. reasoning ones) can return null content
        raise RuntimeError("model returned empty/null content")
    _log.debug("API response: %d chars", len(resp))
    # tolerant parse: recovers fenced / prose-wrapped / trailing-junk payloads
    return loads_lenient(resp)