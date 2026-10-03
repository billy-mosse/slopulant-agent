import hashlib
import numpy as np

D = 128

def _tok(s):
    return [x for x in s.lower().replace(',', ' ').split() if len(x) > 2]

def _vec(p):
    v = np.zeros(D)
    for t in _tok(p['title'] + ' ' + p['desc']):
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        v[h % D] += 1.0 if (h >> 8) & 1 else -1.0
    return v / (np.linalg.norm(v) or 1.0)

def _rank(p, k):
    s = [x['sku'] for x in p]
    m = np.stack([_vec(x) for x in p])
    c = m @ m.T
    out = {}
    for i, sku in enumerate(s):
        idx = np.argsort(-c[i])
        out[sku] = [(s[j], float(c[i, j])) for j in idx if j != i][:k]
    return out

if __name__ == '__main__':
    data = [
        {'sku': 'BED-001', 'title': 'Queen linen duvet cover', 'desc': 'Stonewashed linen, oat'},
        {'sku': 'BED-003', 'title': 'Linen pillowcase set', 'desc': 'Stonewashed linen, oat, set of 2'},
        {'sku': 'BTH-010', 'title': 'Turkish cotton bath towel', 'desc': '600gsm, white'},
    ]
    print(_rank(data, 2))
