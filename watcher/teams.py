"""Team membership and folder ownership, read from teams.yaml at the root of the
monorepo (on the base branch). Missing file = everything is "unassigned"."""
import yaml

from . import gitrepo

_cache = {}


def load(sha):
    """{"teams": {key: {name, lead, members, folders}}, "folder_team": {...}, "person_team": {...}}"""
    if sha in _cache:
        return _cache[sha]
    try:
        data = yaml.safe_load(gitrepo.git("show", f"{sha}:teams.yaml")) or {}
    except Exception:
        data = {}
    teams = data.get("teams") or {}
    out = {
        "teams": teams,
        "folder_team": {f: key for key, t in teams.items() for f in t.get("folders") or []},
        "person_team": {p: key for key, t in teams.items() for p in t.get("members") or []},
    }
    _cache[sha] = out
    return out


def team_name(info, key):
    return (info["teams"].get(key) or {}).get("name", key or "Unassigned") if key else "Unassigned"
