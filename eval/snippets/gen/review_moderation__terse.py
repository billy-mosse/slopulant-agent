import re

def _clean(s):
    s = s.lower()
    s = re.sub(r"http\S+", " <link> ", s)
    return re.sub(r"[^a-z<>\s]", " ", s)

def _check(r):
    t = _clean(r["text"])
    if "<link>" in t: return "spam"
    if any(w in t.split() for w in {"scam","garbage","idiot"}): return "badword"
    alpha = [c for c in r["text"] if c.isalpha()]
    if alpha and sum(c.isupper() for c in alpha)/len(alpha) > 0.7: return "caps"
    return None

def process(items):
    ok, bad = [], []
    for i in items:
        m = _check(i)
        (bad.append((i["review_id"], m)) if m else ok.append(i))
    return ok, bad
