"""Local clone of the monorepo. Each top-level folder is one "repo".
Reads code straight from git objects, so no checkouts are needed."""
import subprocess

from . import config


def git(*args):
    return subprocess.run(
        ["git", "-C", str(config.CLONE_DIR), *args], capture_output=True, text=True, check=True
    ).stdout


def ensure_clone():
    if not (config.CLONE_DIR / ".git").exists():
        config.CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["gh", "repo", "clone", config.GITHUB_REPO, str(config.CLONE_DIR), "--", "--no-checkout"],
            capture_output=True, check=True,
        )


def fetch_base():
    """Fetches the base branch, returns its commit sha."""
    git("fetch", "--quiet", "origin", config.BASE_BRANCH)
    return git("rev-parse", "FETCH_HEAD").strip()


def fetch_pr(pr_number):
    git("fetch", "--quiet", "origin", f"+pull/{pr_number}/head:refs/pr/{pr_number}")


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
        except UnicodeDecodeError:
            continue
    return files
