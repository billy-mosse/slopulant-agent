def shingles(s, n=3):
    s = s.lower()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def related(catalog, k=4):
    """For each product, the k most similar products by title+description text, for the PDP carousel."""
    sh = {p["id"]: shingles(p["title"] + " " + p["description"]) for p in catalog}
    out = {}
    for a, sa in sh.items():
        sims = []
        for b, sb in sh.items():
            if a != b:
                sims.append((b, len(sa & sb) / len(sa | sb)))
        out[a] = sorted(sims, key=lambda x: x[1], reverse=True)[:k]
    return out
