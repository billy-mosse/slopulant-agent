"""All settings come from env vars (or a gitignored .env file next to the repo
root), so the same code runs on a laptop (OpenRouter) and on the hackathon box
(Ollama + OpenClaw): only the .env differs."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """KEY=VALUE lines; real environment variables win. ~ is expanded."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), os.path.expanduser(value.strip().strip('"').strip("'")))


_load_dotenv(ROOT / ".env")

GITHUB_REPO = os.environ.get("GITHUB_REPO", "billy-mosse/slopulant-monorepo")
# Token for GitHub API + git fetch/push: GITHUB_TOKEN, else this file, else (laptop
# only) `gh auth token`. On shared machines set ALLOW_GH_FALLBACK=0 so another
# user's gh login is never used.
GITHUB_TOKEN_FILE = Path(os.environ.get("GITHUB_TOKEN_FILE", Path.home() / ".slopulant" / "github_token"))
ALLOW_GH_FALLBACK = os.environ.get("ALLOW_GH_FALLBACK", "1") != "0"
# Post/update the "[oc]" alert comment on PRs.
POST_GITHUB_COMMENTS = os.environ.get("POST_GITHUB_COMMENTS", "1") != "0"
BASE_BRANCH = os.environ.get("BASE_BRANCH", "main")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))

# LLM profiles: the same model (Qwen3-Coder-Next) served locally (Ollama on the GB10
# box) or by OpenRouter. LLM_PROFILE picks the default; the dashboard can switch at
# runtime (stored in DATA_DIR/llm_profile). Results are cached per model, so
# switching back and forth never throws work away.
def _openrouter_key():
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    key_file = Path(os.environ.get("OPENROUTER_KEY_FILE", ROOT / ".api_key"))  # gitignored
    return key_file.read_text().strip() if key_file.exists() else ""


PROFILES = {
    "local": {
        "label": "Local · Qwen3-Coder-Next Q6 (Ollama)",
        "base_url": os.environ.get("LLM_LOCAL_URL", "http://127.0.0.1:11434/v1"),
        "model": os.environ.get("LLM_LOCAL_MODEL", "coder-next:latest"),
        "api_key": "",
    },
    "openrouter": {
        "label": "OpenRouter · Qwen3-Coder-Next bf16",
        "base_url": "https://openrouter.ai/api/v1",
        "model": os.environ.get("LLM_OPENROUTER_MODEL", "qwen/qwen3-coder-next"),
        "api_key": _openrouter_key(),
    },
}
DEFAULT_PROFILE = os.environ.get("LLM_PROFILE", "openrouter")


def llm_profile():
    """Active profile name: dashboard override if set, else LLM_PROFILE."""
    try:
        name = (DATA_DIR / "llm_profile").read_text().strip()
    except OSError:
        name = ""
    return name if name in PROFILES else DEFAULT_PROFILE


def set_llm_profile(name):
    if name not in PROFILES:
        raise ValueError(f"unknown LLM profile {name!r}; choose from {', '.join(PROFILES)}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "llm_profile").write_text(name)


def llm():
    """{"profile", "label", "base_url", "model", "api_key"} for the active profile."""
    name = llm_profile()
    return {"profile": name, **PROFILES[name]}


# Local cache of LLM responses (JSON): identical requests never hit the provider.
# Turn off for the hackathon deployment with LLM_CACHE=0.
LLM_CACHE = os.environ.get("LLM_CACHE", "1") != "0"
LLM_CACHE_PATH = Path(os.environ.get("LLM_CACHE_PATH", ROOT / "data" / "llm_cache.json"))

# tqdm progress bars (with ETA) for indexing and scoring; PROGRESS=0 to hide.
PROGRESS = os.environ.get("PROGRESS", "1") != "0"

# Parallel LLM requests (OpenRouter and vLLM both handle concurrent requests well).
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "8"))

# Bump when compare()/ranking logic changes: open PRs get re-scored.
SCORING_VERSION = 6

# Bump when the topic prompts or topic contents change; part of the cache key.
TOPICS_VERSION = 3

EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "watcher.db"
CLONE_DIR = DATA_DIR / "clone"

# Per PR folder, the top-K folders (plus any dataflow link) are flagged as
# candidates for the downstream duplicate detector.
# 5: on eval/run_eval.py, K=3 -> 5 lifts recall 0.97 -> 1.00 for +0.3 candidates per
# folder; the floor below keeps unrelated PRs silent either way.
TOP_K = int(os.environ.get("TOP_K", "5"))
# ...but only if they clear this floor. Score = mean(description, keywords). On the
# 185-query eval, 0.38 keeps 97% of true top-3 matches, drops 62% of wrong ones and
# leaves 80% of unrelated systems with no candidate. Retune with eval/prompt_lab.py.
MIN_CANDIDATE_SCORE = float(os.environ.get("MIN_CANDIDATE_SCORE", "0.38"))

# Files bigger than this are skipped entirely (generated/data files).
MAX_FILE_BYTES = 400_000

# Folders with more code than this (chars) are split into chunks, topics are
# extracted per chunk and merged. ~40k chars is ~12k tokens: comfortable inside a
# 32k-token context with the prompt and the JSON reply.
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "40000"))

# Dummy duplicate classifier (to be replaced by the CLM classifier): one LLM call
# per candidate. CLASSIFIER=0 to skip.
CLASSIFIER = os.environ.get("CLASSIFIER", "1") != "0"

# OpenClaw: the agent that writes the alert. If the CLI isn't available (e.g. on a
# laptop) the alert falls back to a deterministic summary.
OPENCLAW_BIN = os.environ.get("OPENCLAW_BIN", "openclaw")
OPENCLAW_AGENT = os.environ.get("OPENCLAW_AGENT", "main")
OPENCLAW_TIMEOUT = int(os.environ.get("OPENCLAW_TIMEOUT", "180"))

# A queue item stuck in "running" longer than this (a crashed run) is retried.
STALE_RUN_SECONDS = int(os.environ.get("STALE_RUN_SECONDS", "900"))

# Re-extract topics for the folders a PR touches on every run, even if that folder
# version was seen before (main's folders always come from the cache).
# REEXTRACT_PR_TOPICS=0 reuses cached PR topics too.
REEXTRACT_PR_TOPICS = os.environ.get("REEXTRACT_PR_TOPICS", "1") != "0"
