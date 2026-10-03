import re
from collections import Counter

A = {"soft", "scratchy", "color", "size", "shrink", "warm", "quality", "price", "pilling"}

def _t(s):
    s = re.sub(r"http\S+", "", s.lower())
    return re.sub(r"[^a-z\s]", " ", s)

def _g(r):
    d = {}
    for x in r:
        d.setdefault(x["sku"], []).append(x)
    return d

def _f(r):
    g = _g(r)
    return {
        sku: {
            "avg_rating": round(sum(x["rating"] for x in rs) / len(rs), 2),
            "summary": f"Customers mention: {', '.join(a for a,_ in Counter(w for x in rs for w in _t(x['text']).split() if w in A).most_common(3)) or 'n/a'}"
        }
        for sku, rs in g.items()
    }
