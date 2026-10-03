"""Local clone of the monorepo. Each top-level folder is one "repo".
Reads code straight from git objects, so no checkouts are needed. Network
operations authenticate with the GitHub token via an http header (never stored in
.git/config, never printed)."""
import subprocess
from collections import Counter

from . import config, github


def _run(args, network=False):
    auth = github.git_auth_args() if network else []
    proc = subprocess.run(["git", *auth, "-C", str(config.CLONE_DIR), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.replace(github.token(), "***") if network else proc.stderr
        raise RuntimeError(f"git {args[0]} failed: {err.strip()[:300]}")
    return proc.stdout


def git(*args):
    return _run(args)


def ensure_clone():
    if (config.CLONE_DIR / ".git").exists():
        return
    config.CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{config.GITHUB_REPO}.git"
    proc = subprocess.run(["git", *github.git_auth_args(), "clone", "--quiet", "--no-checkout", url, str(config.CLONE_DIR)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git clone failed: {proc.stderr.replace(github.token(), '***').strip()[:300]}")


def fetch_base():
    """Fetches the base branch, returns its commit sha."""
    _run(["fetch", "--quiet", "origin", f"+{config.BASE_BRANCH}:refs/remotes/origin/{config.BASE_BRANCH}"], network=True)
    return git("rev-parse", f"refs/remotes/origin/{config.BASE_BRANCH}").strip()


def fetch_pr(pr_number):
    _run(["fetch", "--quiet", "origin", f"+pull/{pr_number}/head:refs/pr/{pr_number}"], network=True)


def fetch_branches():
    """All branches -> refs/remotes/origin/*; returns {branch: sha}."""
    _run(["fetch", "--quiet", "--prune", "origin", "+refs/heads/*:refs/remotes/origin/*"], network=True)
    out = {}
    for line in git("for-each-ref", "--format=%(refname:short) %(objectname)", "refs/remotes/origin").splitlines():
        name, sha = line.split()
        if name.startswith("origin/") and name != "origin/HEAD":
            out[name[len("origin/"):]] = sha
    return out


def force_push_main(sha):
    """Reset the remote base branch to `sha` (demo reset only)."""
    _run(["push", "--quiet", "--force", "origin", f"{sha}:refs/heads/{config.BASE_BRANCH}"], network=True)


def folders(sha):
    """{folder_name: tree_sha} for every top-level folder at a commit."""
    out = {}
    for line in git("ls-tree", sha).splitlines():
        meta, name = line.split("\t", 1)
        _, kind, obj = meta.split()
        if kind == "tree" and not name.startswith("."):
            out[name] = obj
    return out


def touched_folders(base_sha, head_sha):
    """Top-level folders changed between the merge base and the PR head."""
    merge_base = git("merge-base", base_sha, head_sha).strip()
    paths = git("diff", "--name-only", merge_base, head_sha).splitlines()
    return sorted({p.split("/", 1)[0] for p in paths if "/" in p})


def folder_files(sha, folder):
    """[(path, text)] for the files in a folder at a commit, skipping big/binary files."""
    files = []
    for line in git("ls-tree", "-r", "-l", sha, "--", f"{folder}/").splitlines():
        meta, path = line.split("\t", 1)
        _, kind, obj, size = meta.split()
        if kind != "blob" or int(size) > config.MAX_FILE_BYTES:
            continue
        try:
            files.append((path, git("cat-file", "-p", obj)))
        except (UnicodeDecodeError, RuntimeError):
            continue
    return files


def owner(sha, folder):
    """Who to talk to about a folder: the author with the most commits touching it."""
    authors = git("log", "--format=%an", sha, "--", f"{folder}/").splitlines()
    return Counter(authors).most_common(1)[0][0] if authors else None


def commit_author(sha):
    try:
        return git("log", "-1", "--format=%an", sha).strip()
    except RuntimeError:
        return None


def history(base_sha, n=25):
    """Recent first-parent history of main, newest first. Merge commits carry the
    merged branch's commits (second parent side) so the UI can draw the branch."""
    out = []
    fmt = "%H%x00%h%x00%P%x00%an%x00%ct%x00%s"
    for line in git("log", "--first-parent", f"-{n}", f"--format={fmt}", base_sha).splitlines():
        sha, short, parents, author, ts, subject = line.split("\x00")
        parents = parents.split()
        entry = {"sha": sha, "short": short, "author": author, "ts": int(ts), "subject": subject,
                 "parents": parents, "merged": []}
        if len(parents) > 1:  # commits brought in by the merge
            for side in git("log", f"--format={fmt}", f"{parents[0]}..{parents[1]}").splitlines():
                s, sh, p, a, t, subj = side.split("\x00")
                entry["merged"].append({"sha": s, "short": sh, "author": a, "ts": int(t), "subject": subj})
        out.append(entry)
    return out


def merge_base(a, b):
    try:
        return git("merge-base", a, b).strip()
    except RuntimeError:
        return None


def branch_commits(base_sha, head_sha):
    """Commits on a branch that aren't on base (newest first)."""
    fmt = "%H%x00%h%x00%an%x00%ct%x00%s"
    out = []
    for line in git("log", f"--format={fmt}", f"{base_sha}..{head_sha}").splitlines():
        s, sh, a, t, subj = line.split("\x00")
        out.append({"sha": s, "short": sh, "author": a, "ts": int(t), "subject": subj})
    return out
