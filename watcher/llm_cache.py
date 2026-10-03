"""Local cache of LLM responses: identical requests are never sent twice.

A JSON file mapping sha256(request) -> response text. The key covers everything
that affects the answer (endpoint, model, prompt, max_tokens, temperature), so
changing any of them is a miss. Toggle with LLM_CACHE=0."""
import hashlib
import json
import os
import threading

from . import config

_lock = threading.Lock()
_data = None


def _load():
    global _data
    if _data is None:
        path = config.LLM_CACHE_PATH
        _data = json.loads(path.read_text()) if path.exists() else {}
    return _data


def key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def get(request):
    if not config.LLM_CACHE:
        return None
    with _lock:
        return _load().get(key(request))


def put(request, text):
    if not config.LLM_CACHE:
        return
    with _lock:
        data = _load()
        data[key(request)] = text
        path = config.LLM_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)  # atomic: a crash mid-write never corrupts the cache
