"""Validate the small, explicit contract between the flagger and Discord."""

import re
from urllib.parse import urlsplit


def _text(payload, key, *, default=None, limit=6000):
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{key} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _url(value, key):
    if not isinstance(value, str) or len(value) > 2000:
        raise ValueError(f"{key} must be an HTTP(S) URL")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"https", "http"} or not parsed.netloc
            or parsed.username or parsed.password or any(c.isspace() for c in value)):
        raise ValueError(f"{key} must be an HTTP(S) URL without embedded credentials")
    return value


def validate_decision(payload):
    """Return a normalized decision; reject truthy strings such as 'false'."""
    if not isinstance(payload, dict):
        raise ValueError("Decision must be a JSON object")
    decision = {}
    for key in ("decision_id", "repository", "pr_title", "author_login", "topic"):
        decision[key] = _text(payload, key, limit=500 if key == "pr_title" else 256)
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", decision["decision_id"]):
        raise ValueError("decision_id must use letters, numbers, dots, underscores, colons or hyphens")
    decision["head_sha"] = payload.get("head_sha")
    if decision["head_sha"] is not None and (not isinstance(decision["head_sha"], str) or not re.fullmatch(r"[a-fA-F0-9]{7,64}", decision["head_sha"])):
        raise ValueError("head_sha must be the analyzed Git commit hash (7–64 hexadecimal characters), or null")
    source_revision = payload.get("source_revision")
    if source_revision is not None:
        if not isinstance(source_revision, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", source_revision):
            raise ValueError("source_revision must be a 64-character database snapshot fingerprint")
        decision["source_revision"] = source_revision
    if decision["head_sha"] is None and source_revision is None:
        raise ValueError("Supply head_sha or source_revision to identify the analyzed snapshot")
    number = payload.get("pr_number")
    if type(number) is not int or number <= 0:
        raise ValueError("pr_number must be a positive integer")
    decision["pr_number"] = number
    decision["pr_url"] = _url(payload["pr_url"], "pr_url") if payload.get("pr_url") is not None else None
    if type(payload.get("is_duplicate")) is not bool:
        raise ValueError("is_duplicate must be a JSON boolean: true or false")
    decision["is_duplicate"] = payload["is_duplicate"]
    for key, default, choices in (
        ("status", "completed", {"pending", "completed", "failed"}),
        ("pr_state", "open", {"open", "closed", "merged"}),
        ("duplicate_kind", "partial", {"whole_project", "partial", "unspecified"}),
    ):
        value = payload.get(key, default)
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{key} must be one of {', '.join(sorted(choices))}")
        decision[key] = value
    reason = payload.get("reason", "")
    if not isinstance(reason, str) or len(reason) > 6000:
        raise ValueError("reason must be a string of at most 6000 characters")
    if decision["is_duplicate"] and decision["status"] == "completed" and not reason.strip():
        raise ValueError("A completed duplicate decision needs an explanation in reason")
    decision["reason"] = reason.strip()
    for key in ("overlap", "next_step", "note_by"):  # the drafted alert text (watcher source)
        if payload.get(key) is not None:
            decision[key] = _text(payload, key, limit=1000)
    for key in ("author_name", "model_version"):
        if payload.get(key) is not None:
            decision[key] = _text(payload, key, limit=256)
    if payload.get("pr_description") is not None:
        decision["pr_description"] = _text(payload, "pr_description", limit=6000)
    for key in ("source_dup_ids", "source_pr_topic_ids"):
        if key in payload:
            values = payload[key]
            if not isinstance(values, list) or len(values) > 100 or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{key} must be a list of nonnegative integer source IDs")
            decision[key] = sorted(set(values))
    if "source_decision_keys" in payload:
        values = payload["source_decision_keys"]
        if not isinstance(values, list) or len(values) > 100 or any(not isinstance(v, str) or not v.strip() or len(v) > 600 for v in values):
            raise ValueError("source_decision_keys must be a list of '<pr folder>/<existing folder>' keys")
        decision["source_decision_keys"] = sorted(set(values))
    if payload.get("base_sha") is not None:
        if not isinstance(payload["base_sha"], str) or not re.fullmatch(r"[a-fA-F0-9]{7,64}", payload["base_sha"]):
            raise ValueError("base_sha must be a Git commit hash")
        decision["base_sha"] = payload["base_sha"]
    if "folder_names" in payload:
        values = payload["folder_names"]
        if not isinstance(values, list) or len(values) > 100 or any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in values):
            raise ValueError("folder_names must be a list of PR folder names")
        decision["folder_names"] = sorted(set(values))
    if payload.get("discord_user_id") is not None:
        user_id = str(payload["discord_user_id"])
        if not re.fullmatch(r"[0-9]{15,22}", user_id):
            raise ValueError("discord_user_id must be a numeric Discord user ID")
        decision["discord_user_id"] = user_id
    matches = payload.get("matches", [])
    if not isinstance(matches, list) or len(matches) > 25:
        raise ValueError("matches must be a list of at most 25 existing projects/components")
    normalized_matches = []
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("Each match must be an object")
        normalized = {"project_name": _text(match, "project_name", limit=256)}
        for key in ("source_dup_id", "source_topic_id"):
            if key in match:
                if type(match[key]) is not int or match[key] < 0:
                    raise ValueError(f"matches.{key} must be a nonnegative integer")
                normalized[key] = match[key]
        for key in ("topic", "evidence"):
            if match.get(key):
                normalized[key] = _text(match, key, limit=6000)
        if match.get("url"):
            normalized["url"] = _url(match["url"], "matches.url")
        for key in ("pr_files", "existing_files"):
            values = match.get(key, [])
            if (not isinstance(values, list) or len(values) > 100
                    or any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in values)):
                raise ValueError(f"matches.{key} must be a list of file paths")
            normalized[key] = values
        normalized_matches.append(normalized)
    if decision["is_duplicate"] and decision["status"] == "completed" and not normalized_matches:
        raise ValueError("A completed duplicate decision needs at least one existing project/component match")
    decision["matches"] = normalized_matches
    return decision
