import argparse, math, re, logging
from collections import Counter, defaultdict
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ret_rsn")

RS = ("TOO_SMALL", "COLOR_DIFFERENT", "DAMAGED", "QUALITY", "CHANGED_MIND", "LATE")
ALPHA = 1.0
TH = 0.55

PATTERNS = {
    "DAMAGED": [r"\b(arrived|came) (broken|torn|ripped|cracked|damaged)\b", r"\bshattered\b", r"\bstain(ed|s)? (on|out of) the box\b"],
    "LATE": [r"\b(arrived|came|delivered) (too )?late\b", r"\bmissed (the|my) (event|date)\b", r"\bnever arrived on time\b"],
    "TOO_SMALL": [r"\btoo (small|short|narrow)\b", r"\bdoesn'?t fit (my|the) (bed|mattress)\b"],
    "COLOR_DIFFERENT": [r"\b(colou?r|shade) (is |was )?(different|off|not the same)\b"],
}

STOP = {"the","a","an","and","it","is","was","i","to","of","for","my","this","that","in","on","with","but","so","me","they","be","at","as"}

def _clean(s):
    if not s: return ""
    s = s.lower()
    s = re.sub(r"https?://\S+|\S+@\S+|\border\s*#?\s*\d+\b", " ", s)
    s = re.sub(r"[^a-z' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _tok(s):
    t = [x.strip("'") for x in s.split() if x not in STOP and len(x)>1]
    return t + [f"{a}_{b}" for a,b in zip(t,t[1:])]

def _rule(s):
    for r,pats in PATTERNS.items():
        if any(re.search(p,s) for p in pats): return r
    return None

class NB:
    def __init__(s,α=ALPHA): s.α=α; s.p= {}; s.l= {}; s.u= {}; s.v=set()
    def fit(s, X, y):
        cnt = Counter(y)
        n = len(y)
        V = len(s.v)
        cDoc = defaultdict(Counter)
        for t,y_ in zip(X,y): cDoc[y_].update(t); s.v.update(t)
        V = len(s.v)
        for c in RS:
            s.p[c]=math.log((cnt[c]+1)/(n+len(RS)))
            tot = sum(cDoc[c].values())+s.α*V
            s.l[c]={w:math.log((k+s.α)/tot) for w,k in cDoc[c].items()}
            s.u[c]=math.log(s.α/tot)
        return s
    def pred(s, t):
        t = [x for x in t if x in s.v]
        sc = {c:s.p[c]+sum(s.l[c].get(w,s.u[c]) for w in t) for c in RS}
        m = max(sc.values())
        z = sum(math.exp(v-m) for v in sc.values())
        return {c:math.exp(v-m)/z for c,v in sc.items()}

def run(df,nb):
    out = []
    for _,r in df.iterrows():
        c = _clean(r.comment)
        if not c: out.append((r.return_id,"UNKNOWN",0.0,"empty")); continue
        rl = _rule(c)
        if rl: out.append((r.return_id,rl,1.0,"rule")); continue
        probs = nb.pred(_tok(c))
        b = max(probs,key=probs.get)
        if probs[b]<TH: out.append((r.return_id,"UNKNOWN",probs[b],"nb_low_conf"))
        else: out.append((r.return_id,b,probs[b],"nb"))
    return pd.DataFrame(out,columns=["return_id","reason_code","confidence","method"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    eng = create_engine(args.dsn)
    q = text("SELECT return_id, comment, verified_reason, created_at FROM returns.return_requests WHERE created_at >= :since")
    df = pd.read_sql(q, eng, params={"since":args.since})
    train = df[df.verified_reason.isin(RS)]
    nb = NB().fit([_tok(_clean(c)) for c in train.comment], train.verified_reason.tolist())
    log.info("trained on %d, vocab=%d", len(train), len(nb.v))
    lbls = run(df[df.verified_reason.isna()], nb)
    log.info("labels:\n%s", lbls.reason_code.value_counts().to_string())
    if not args.dry:
        lbls.to_sql("reason_labels", eng, schema="returns", if_exists="append", index=False)

if __name__ == "__main__": main()
