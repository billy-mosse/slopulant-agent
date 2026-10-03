"""Technical keywords per system: a separate LLM pass, then programmatic cleanup.

Keywords are compared lexically (TF-IDF over keyword tokens, IDF fitted on the
folders on main), which complements the dense description embedding: a shared rare
term like "jaccard" or "alias map" counts a lot, generic terms count for nothing."""
import math
import re
from collections import Counter

KEYWORDS_PROMPT = """List the technical keywords that identify what the code below does, so it can
be matched with other code doing the same job even if written differently.

Give 10-15 keywords, comma-separated, lowercase, each 1-3 words. Include:
- the specific task (e.g. "near-duplicate detection", "tag normalization")
- algorithms, techniques, similarity measures, model types
- the domain entities it handles (e.g. reviews, skus, sessions, merchants)
- the signals or features it uses, and what it outputs

Do not include generic terms (machine learning, ml, ai, model, data, code, python,
function, system, pipeline, algorithm, e-commerce, business), library names, or
variable/function names. Output only the comma-separated list.

{code}"""

# Dropped whole if a keyword normalizes to one of these.
GENERIC_KEYWORDS = {
    "machine learning", "ml", "ai", "artificial intelligence", "model", "models", "data",
    "dataset", "code", "python", "function", "system", "pipeline", "algorithm", "ecommerce",
    "e commerce", "business", "numpy", "pandas", "sklearn", "scikit learn", "script", "module",
    "analysis", "processing", "input", "output", "inputs", "outputs", "dict", "dictionary",
    "list", "table", "tables", "feature", "features", "score", "scoring", "prediction",
}

# Dropped from keyword tokens before matching.
STOP_TOKENS = {
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "with", "by", "from", "per",
    "based", "using", "via", "into", "data", "ml", "ai", "model", "system",
}


def _stem(token):
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize(keyword):
    k = keyword.lower().strip().strip("`'\".")
    k = re.sub(r"[-_/]", " ", k)
    k = re.sub(r"[^a-z0-9 ]", "", k)
    return " ".join(k.split())


def parse(text):
    """LLM output -> cleaned, de-duplicated keyword list."""
    out = []
    for raw in re.split(r"[,\n;]", text):
        k = normalize(re.sub(r"^\s*[-*\d.]+\s+", "", raw))
        if k and k not in GENERIC_KEYWORDS and len(k) <= 40 and k not in out:
            out.append(k)
    return out


def tokens(keywords):
    return [_stem(t) for k in keywords for t in k.split() if t not in STOP_TOKENS and len(t) > 1]


class TfidfIndex:
    """IDF fitted on a reference set (folders on main); cosine between keyword docs."""

    def __init__(self, reference_docs):
        self.n = len(reference_docs)
        self.df = Counter(t for doc in reference_docs for t in set(tokens(doc)))

    def vector(self, keywords):
        tf = Counter(tokens(keywords))
        v = {t: c * (math.log((self.n + 1) / (self.df.get(t, 0) + 1)) + 1) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def similarity(self, a, b):
        va, vb = self.vector(a), self.vector(b)
        return sum(x * vb.get(t, 0.0) for t, x in va.items())

    def shared(self, a, b):
        """Keywords of `a` that share a token with `b`, rarest (most telling) first."""
        tb = set(tokens(b))
        weight = lambda k: max((math.log((self.n + 1) / (self.df.get(t, 0) + 1)) for t in tokens([k]) if t in tb), default=0)
        return sorted((k for k in a if set(tokens([k])) & tb), key=weight, reverse=True)
