import hashlib
import numpy as np

D = 128

def tok(s):
    return [w for w in s.lower().replace(',', ' ').split() if len(w) > 2]

def enc(tokens):
    v = np.zeros(D)
    for x in tokens:
        h = int(hashlib.md5(x.encode()).hexdigest(), 16)
        v[h % D] += 1 if (h >> 8) & 1 else -1
    n = np.linalg.norm(v)
    return v / n if n else v

def mk(src, tagmap):
    return {r['sku']: enc(tok(r['title']) + tok(r['desc']) + [f't:{z}' for z in tagmap.get(r['sku'], [])]) for r in src}

if __name__ == '__main__':
    src = [
        {'sku': 'BED-001', 'title': 'Queen linen duvet cover', 'desc': 'Stonewashed linen, oat'},
        {'sku': 'BED-002', 'title': 'Queen linen duvet', 'desc': 'Stone washed linen in oat color'},
    ]
    tagmap = {'BED-001': ['bedding', 'linen'], 'BED-002': ['bedding', 'linen']}
    res = mk(src, tagmap)
    print({k: v[:4].round(3).tolist() for k, v in res.items()})
