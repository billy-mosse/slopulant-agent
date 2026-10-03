"""Open a fresh PR from a copy of a test branch, for a clean demo recording.

GitHub never lets you delete a PR's "closed / reopened" timeline events, so a test PR that
was opened and closed several times keeps that history. This pushes the branch's commit to
a new branch `demo/<branch>-<MMDD-HHMM>` and opens a new PR from it (same title, same code,
clean timeline, new PR number, so it also gets a fresh Discord alert).

    .venv/bin/python scripts/fresh_demo_pr.py leo/review-ring-detector       # open
    .venv/bin/python scripts/fresh_demo_pr.py leo/review-ring-detector --dry-run
    .venv/bin/python scripts/fresh_demo_pr.py --cleanup    # close demo/* PRs, delete demo/* branches
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watcher import config, github, gitrepo  # noqa: E402

PREFIX = "demo/"


def fresh(branch, dry_run=False):
    gitrepo.ensure_clone()
    branches = gitrepo.fetch_branches()
    if branch not in branches:
        sys.exit(f"no branch {branch!r}; test branches: {', '.join(sorted(b for b in branches if b != config.BASE_BRANCH))}")
    sha = branches[branch]
    prs = [p for p in github.all_prs() if p["branch"] == branch]
    title = max(prs, key=lambda p: p["number"])["title"] if prs else branch
    new = f"{PREFIX}{branch.replace('/', '-')}-{time.strftime('%m%d-%H%M')}"
    print(f"{branch} @ {sha[:7]} -> branch {new}, PR title {title!r}")
    if dry_run:
        return None
    gitrepo._run(["push", "--quiet", "origin", f"{sha}:refs/heads/{new}"], network=True)
    pr = github.create_pr(new, title, "Demo PR for the overlap watcher.")
    print(f"opened PR #{pr['number']}: {pr['url']}")
    return pr


def cleanup():
    gitrepo.ensure_clone()
    for pr in github.open_prs():
        if pr["branch"].startswith(PREFIX):
            github.close_pr(pr["number"])
            print(f"closed #{pr['number']} ({pr['branch']})")
    for branch in gitrepo.fetch_branches():
        if branch.startswith(PREFIX):
            gitrepo._run(["push", "--quiet", "origin", f":refs/heads/{branch}"], network=True)
            print(f"deleted branch {branch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        cleanup()
    elif args.branch:
        fresh(args.branch, args.dry_run)
    else:
        parser.error("give a branch, or --cleanup")


if __name__ == "__main__":
    main()
