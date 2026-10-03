"""Polls GitHub for open PRs. Uses ETags so unchanged polls return 304 and
don't count against the rate limit."""
import os
import subprocess

import requests

from . import config

_etag = None
_last = []


def _token():
    return os.environ.get("GITHUB_TOKEN") or subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True, check=True
    ).stdout.strip()


def open_prs():
    """Returns [{'number': int, 'head_sha': str, 'title': str, 'author': str}]."""
    global _etag, _last
    headers = {"Authorization": f"Bearer {_token()}", "Accept": "application/vnd.github+json"}
    if _etag:
        headers["If-None-Match"] = _etag
    resp = requests.get(
        f"https://api.github.com/repos/{config.GITHUB_REPO}/pulls",
        params={"state": "open", "base": config.BASE_BRANCH, "per_page": 100},
        headers=headers,
        timeout=20,
    )
    if resp.status_code == 304:
        return _last
    resp.raise_for_status()
    _etag = resp.headers.get("ETag")
    _last = [
        {
            "number": pr["number"],
            "head_sha": pr["head"]["sha"],
            "title": pr["title"],
            "author": pr["user"]["login"],
            "branch": pr["head"]["ref"],
            "url": pr["html_url"],
        }
        for pr in resp.json()
    ]
    return _last
