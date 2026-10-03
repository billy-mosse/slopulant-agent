"""Demo controls for the dashboard: open / merge / close real PRs, save a baseline,
and reset the repo to it.

Baseline = main's commit + which test branches have an open PR. Reset force-pushes
main back, re-opens the baseline PRs (merged ones come back as new PRs from the
same branch: GitHub can't un-merge), closes PRs opened during the demo, and clears
demo results from the DB. Cached topics are kept: they're keyed by content."""
import json
import time

from . import config, db, github, gitrepo


def _latest_pr_per_branch(prs):
    latest = {}
    for pr in sorted(prs, key=lambda p: p["number"]):
        latest[pr["branch"]] = pr
    return latest


def catalog():
    """Every branch except main, with its latest PR (if any) and whether it's open."""
    branches = gitrepo.fetch_branches()
    prs = github.all_prs()
    latest = _latest_pr_per_branch(prs)
    out = []
    for branch, sha in sorted(branches.items()):
        if branch == config.BASE_BRANCH:
            continue
        pr = latest.get(branch)
        out.append({
            "branch": branch, "head_sha": sha, "title": pr["title"] if pr else branch,
            "pr_number": pr["number"] if pr else None, "state": pr["state"] if pr else "no PR",
            "author": gitrepo.commit_author(sha),
        })
    return out


def open_branch(branch):
    """Opens a PR for a branch: reopens its closed PR if possible, else creates one."""
    prs = [p for p in github.all_prs() if p["branch"] == branch]
    if any(p["state"] == "open" for p in prs):
        return next(p for p in prs if p["state"] == "open"), "already open"
    reopenable = [p for p in prs if p["state"] == "closed"]
    if reopenable:
        return github.reopen_pr(max(reopenable, key=lambda p: p["number"])["number"]), "reopened"
    title = max(prs, key=lambda p: p["number"])["title"] if prs else branch
    return github.create_pr(branch, title, "Demo PR for the overlap watcher."), "created"


def merge(number):
    return github.merge_pr(number)


def close(number):
    return github.close_pr(number)


def save_baseline(conn):
    main_sha = gitrepo.fetch_base()
    open_branches = sorted(p["branch"] for p in github.open_prs())
    baseline = {"main_sha": main_sha, "open_branches": open_branches, "saved_at": time.time()}
    db.set_kv(conn, "baseline", json.dumps(baseline))
    db.add_event(conn, "baseline", f"baseline saved: main @ {main_sha[:7]}, {len(open_branches)} open PRs")
    return baseline


def baseline(conn):
    raw = db.get_kv(conn, "baseline")
    return json.loads(raw) if raw else None


def reset(conn):
    base = baseline(conn)
    if not base:
        raise RuntimeError("no baseline saved yet")
    steps = []
    current = gitrepo.fetch_base()
    if current != base["main_sha"]:
        gitrepo.force_push_main(base["main_sha"])
        steps.append(f"main {current[:7]} → {base['main_sha'][:7]} (force-push)")
    open_now = {p["branch"]: p for p in github.open_prs()}
    for branch, pr in open_now.items():
        if branch not in base["open_branches"]:
            github.close_pr(pr["number"])
            steps.append(f"closed #{pr['number']} ({branch})")
    for branch in base["open_branches"]:
        if branch not in open_now:
            pr, how = open_branch(branch)
            steps.append(f"{how} #{pr['number']} ({branch})")
    # Demo results go; cached folders/topics stay (keyed by content, still valid).
    for table in ("scores", "decisions", "alerts", "pr_queue", "events"):
        conn.execute(f"DELETE FROM {table}")
    # clm_finished markers go too: otherwise a re-scored commit never gets its note and its
    # Discord alert stays pending.
    conn.execute("DELETE FROM kv WHERE key LIKE 'indexed_base_sha%' OR key = 'last_error' OR key LIKE 'clm_finished:%'")
    conn.commit()
    db.add_event(conn, "reset", "reset to baseline: " + ("; ".join(steps) or "nothing to change"), steps=steps)
    return steps
