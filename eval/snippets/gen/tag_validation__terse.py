VOC = {"bedding", "bath", "linen", "cotton", "towel", "duvet", "pillow", "decor", "kitchen"}

AL = {"linens": "linen", "linen-blend": "linen", "towels": "towel", "pillows": "pillow", "bed": "bedding", "home-decor": "decor"}

def nrm(t):
    return AL.get(t.strip().lower().replace(" ", "-"), t.strip().lower().replace(" ", "-"))

def cls(r):
    o = {}
    for s, t in r:
        x = nrm(t)
        if x in VOC:
            o.setdefault(s, set()).add(x)
    return {s: sorted(v) for s, v in o.items()}
