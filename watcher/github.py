"""GitHub API for the watcher and the demo dashboard.

Polling uses ETags, so unchanged polls return 304 and don't count against the rate
limit. The token comes from GITHUB_TOKEN, else GITHUB_TOKEN_FILE, else (laptop only)
`gh auth token`; it is never logged."""
import base64
import os
import subprocess

import requests

from . import config

API = "https://api.github.com"
COMMENT_PREFIX = "[oc]"

_etag = None
_last = []
_token_cache = None
_login_cache = None


def token():
    global _token_cache
    if _token_cache:
        return _token_cache
    if os.environ.get("GITHUB_TOKEN"):
        _token_cache = os.environ["GITHUB_TOKEN"].strip()
    elif config.GITHUB_TOKEN_FILE.exists():
        _token_cache = config.GITHUB_TOKEN_FILE.read_text().strip()
    elif config.ALLOW_GH_FALLBACK:
        _token_cache = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    else:
        raise RuntimeError(f"no GitHub token: set GITHUB_TOKEN or create {config.GITHUB_TOKEN_FILE}")
    return _token_cache


_token = token  # backwards-compatible name used by the eval scripts


def git_auth_args():
    """`git -c ...` args that authenticate https fetch/push with the token, without
    storing it in .git/config or printing it."""
    basic = base64.b64encode(f"x-access-token:{token()}".encode()).decode()
    return ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]


def _headers(extra=None):
    return {"Authorization": f"Bearer {token()}", "Accept": "application/vnd.github+json", **(extra or {})}


def api(method, path, **kwargs):
    resp = requests.request(method, f"{API}{path}", headers=_headers(kwargs.pop("headers", None)), timeout=30, **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(f"GitHub {method} {path} -> {resp.status_code}: {resp.text[:300]}")
    return resp.json() if resp.content else None


def _pr_summary(pr):
    return {
        "number": pr["number"],
        "head_sha": pr["head"]["sha"],
        "title": pr["title"],
        "author": pr["user"]["login"],
        "branch": pr["head"]["ref"],
        "url": pr["html_url"],
        "state": "merged" if pr.get("merged_at") else pr["state"],
        "merged_at": pr.get("merged_at"),
        "merge_sha": pr.get("merge_commit_sha") if pr.get("merged_at") else None,
    }


def open_prs():
    """Open PRs against the base branch (ETag-cached)."""
    global _etag, _last
    headers = {"If-None-Match": _etag} if _etag else {}
    resp = requests.get(
        f"{API}/repos/{config.GITHUB_REPO}/pulls",
        params={"state": "open", "base": config.BASE_BRANCH, "per_page": 100},
        headers=_headers(headers),
        timeout=20,
    )
    if resp.status_code == 304:
        return _last
    resp.raise_for_status()
    _etag = resp.headers.get("ETag")
    _last = [_pr_summary(pr) for pr in resp.json()]
    return _last


def recently_closed_prs(limit=30):
    """Most recently updated closed PRs (merged or not), newest first."""
    prs = api("GET", f"/repos/{config.GITHUB_REPO}/pulls",
              params={"state": "closed", "base": config.BASE_BRANCH, "sort": "updated", "direction": "desc", "per_page": limit})
    return [_pr_summary(pr) for pr in prs]


def all_prs():
    prs, page = [], 1
    while True:
        batch = api("GET", f"/repos/{config.GITHUB_REPO}/pulls", params={"state": "all", "per_page": 100, "page": page})
        prs += [_pr_summary(pr) for pr in batch]
        if len(batch) < 100:
            return prs
        page += 1


def branches():
    out, page = [], 1
    while True:
        batch = api("GET", f"/repos/{config.GITHUB_REPO}/branches", params={"per_page": 100, "page": page})
        out += [b["name"] for b in batch]
        if len(batch) < 100:
            return out
        page += 1


def create_pr(branch, title, body=""):
    return _pr_summary(api("POST", f"/repos/{config.GITHUB_REPO}/pulls",
                           json={"title": title, "head": branch, "base": config.BASE_BRANCH, "body": body}))


def reopen_pr(number):
    return _pr_summary(api("PATCH", f"/repos/{config.GITHUB_REPO}/pulls/{number}", json={"state": "open"}))


def close_pr(number):
    return _pr_summary(api("PATCH", f"/repos/{config.GITHUB_REPO}/pulls/{number}", json={"state": "closed"}))


def merge_pr(number, title=None):
    """Merge commit (not squash), so the PR's folder trees land on main unchanged
    and their cached topics are reused as-is."""
    body = {"merge_method": "merge"}
    if title:
        body["commit_title"] = title
    return api("PUT", f"/repos/{config.GITHUB_REPO}/pulls/{number}/merge", json=body)


def login():
    global _login_cache
    if _login_cache is None:
        _login_cache = api("GET", "/user")["login"]
    return _login_cache


def upsert_alert_comment(number, body):
    """Creates or edits the single "[oc]" comment on a PR. Returns (url, changed)."""
    if not body.startswith(COMMENT_PREFIX):
        body = f"{COMMENT_PREFIX} {body}"
    comments = api("GET", f"/repos/{config.GITHUB_REPO}/issues/{number}/comments", params={"per_page": 100})
    mine = [c for c in comments if c["body"].startswith(COMMENT_PREFIX) and c["user"]["login"] == login()]
    if mine:
        if mine[0]["body"] == body:
            return mine[0]["html_url"], False
        c = api("PATCH", f"/repos/{config.GITHUB_REPO}/issues/comments/{mine[0]['id']}", json={"body": body})
        return c["html_url"], True
    c = api("POST", f"/repos/{config.GITHUB_REPO}/issues/{number}/comments", json={"body": body})
    return c["html_url"], True
