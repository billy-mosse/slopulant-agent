"""Local cache of LLM responses: identical requests are never sent twice.

A JSON file mapping sha256(request) -> response text. The key covers everything
that affects the answer (endpoint, model, prompt, max_tokens, temperature), so
changing any of them is a miss. Toggle with LLM_CACHE=0.

Several processes (the demo server, an eval run) can share the file: writes take an
exclusive file lock and merge with what's on disk, so nobody's entries get lost."""
import fcntl
import hashlib
import json
import os
import threading
from contextlib import contextmanager

from . import config

_lock = threading.Lock()
_data = None


def _read():
    path = config.LLM_CACHE_PATH
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


@contextmanager
def _file_lock():
    path = config.LLM_CACHE_PATH.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def get(request):
    global _data
    if not config.LLM_CACHE:
        return None
    k = key(request)
    with _lock:
        if _data is None:
            _data = _read()
        if k not in _data:
            _data.update(_read())  # another process may have answered it meanwhile
        return _data.get(k)


def put(request, text):
    global _data
    if not config.LLM_CACHE:
        return
    with _lock, _file_lock():
        data = _read()  # merge: keep entries other processes wrote since we loaded
        data.update(_data or {})
        data[key(request)] = text
        path = config.LLM_CACHE_PATH
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)  # atomic: a crash mid-write never corrupts the cache
        _data = data
