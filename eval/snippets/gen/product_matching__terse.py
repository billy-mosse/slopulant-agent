from difflib import SequenceMatcher
from itertools import combinations

T = 0.85

def norm(t):
    return " ".join(t.lower().replace("-", " ").split())

def sim(x, y):
    if x["brand"] != y["brand"]:
        return 0.0
    ts = SequenceMatcher(None, norm(x["title"]), norm(y["title"])).ratio()
    ps = min(x["price"], y["price"]) / max(x["price"], y["price"])
    return 0.8 * ts + 0.2 * ps

def dupes(items):
    res = []
    for i, j in combinations(items, 2):
        s = sim(i, j)
        if s >= T:
            res.append((i["sku"], j["sku"], round(s, 3)))
    return res

if __name__ == "__main__":
    items = [
        {"sku": "BED-001", "title": "Queen Linen Duvet Cover - Oat", "brand": "Hearth", "price": 189.0},
        {"sku": "BED-002", "title": "Queen linen duvet cover oat", "brand": "Hearth", "price": 179.0},
        {"sku": "BTH-010", "title": "Turkish Cotton Bath Towel", "brand": "Loom", "price": 34.0},
    ]
    print(dupes(items))
