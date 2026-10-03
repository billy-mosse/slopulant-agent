# misc helpers

CANON = {"bed", "bath", "linen", "cotton", "towel", "duvet", "pillow", "decor", "kitchen"}
SYN = {"linens": "linen", "towels": "towel", "pillows": "pillow", "bedding": "bed", "homedecor": "decor"}


def fix(x):
    x = x.lower().strip().replace("-", "").replace(" ", "")
    return SYN.get(x, x)


def process(rows):
    out = {}
    for item, label in rows:
        v = fix(label)
        if v in CANON:
            out.setdefault(item, []).append(v)
    return {k: sorted(set(v)) for k, v in out.items()}
