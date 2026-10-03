"""All settings come from env vars so the same code runs on a Mac (mlx_lm.server)
and on the hackathon box (vLLM) — only LLM_BASE_URL / LLM_MODEL change."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GITHUB_REPO = os.environ.get("GITHUB_REPO", "billy-mosse/slopulant-monorepo")
BASE_BRANCH = os.environ.get("BASE_BRANCH", "main")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))

# Any OpenAI-compatible server. Default: Qwen3-Coder-Next on OpenRouter (dev).
# At the hackathon, serve the same model locally with vLLM and set
# LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=<served model name>.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3-coder-next")


def _api_key():
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    key_file = ROOT / ".api_key"  # gitignored
    return key_file.read_text().strip() if key_file.exists() else ""


# Only sent to hosted endpoints; local servers don't need it.
LLM_API_KEY = _api_key() if "openrouter.ai" in LLM_BASE_URL else os.environ.get("LLM_API_KEY", "")

# Local cache of LLM responses (JSON): identical requests never hit the provider.
# Turn off for the hackathon deployment with LLM_CACHE=0.
LLM_CACHE = os.environ.get("LLM_CACHE", "1") != "0"
LLM_CACHE_PATH = Path(os.environ.get("LLM_CACHE_PATH", ROOT / "data" / "llm_cache.json"))

# tqdm progress bars (with ETA) for indexing and scoring; PROGRESS=0 to hide.
PROGRESS = os.environ.get("PROGRESS", "1") != "0"

# Parallel LLM requests (OpenRouter and vLLM both handle concurrent requests well).
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "8"))

# Bump when compare()/ranking logic changes: open PRs get re-scored.
SCORING_VERSION = 4

# Bump when the card prompts or card contents change; part of the cache key.
CARDS_VERSION = 5

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

# Files bigger than this are skipped when describing a folder.
MAX_FILE_BYTES = 40_000
