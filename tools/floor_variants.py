"""Calibrate the content-floor text recipe against the published 0.1565 / 0.3377."""
import gzip, json, sys, time
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

VARIANTS = {
    "readme1500_sublinear": dict(readme=1500, sublinear=True, tok=r"(?u)\b\w+\b"),
    "readme0_sublinear":    dict(readme=0,    sublinear=True, tok=r"(?u)\b\w+\b"),
    "readme500_sublinear":  dict(readme=500,  sublinear=True, tok=r"(?u)\b\w+\b"),
    "readme1500_plain_tf":  dict(readme=1500, sublinear=False, tok=r"(?u)\b\w+\b"),
    "readme1500_default_tok": dict(readme=1500, sublinear=True, tok=r"(?u)\b\w\w+\b"),
}
DATASETS = sys.argv[1:] or ["npm"]
SEEDS = [101,102,103,104,105,106,107,108,109,110]
F_TEST, F_VAL, JIT = 0.07, 0.08, 0.05

def load(ds):
    p = Path(__file__).resolve().parent.parent / "data" / "graphs" / f"{ds}_graph.json.gz"
    with gzip.open(p, "rb") as fh: payload = json.loads(fh.read().decode())
    nodes = payload["nodes"]
    return nodes, np.array(payload["edges"], np.int64)

def split(days, seed):
    n=len(days); order=np.argsort(days,kind="stable"); n_test=max(1,round(F_TEST*n))
    rng=np.random.default_rng(seed*7919+13); n_val=max(1,round(F_VAL*(1+JIT*(2*rng.random()-1))*n))
    start=n-n_test
    return np.sort(order[:max(0,start-n_val)]), np.sort(order[max(0,start-n_val):start]), np.sort(order[start:])

def evaluate(Z, src, tgt, idx, cand):
    mask=np.zeros(len(Z),bool); mask[idx]=True; sel=mask[src]
    pos={}
    for a,b in zip(src[sel].tolist(), tgt[sel].tolist()): pos.setdefault(a,[]).append(b)
    rr=[]
    for a,targets in pos.items():
        keep=cand[cand!=a]; sc=Z[keep]@Z[a]
        rk={int(keep[i]):r+1 for r,i in enumerate(np.argsort(-sc,kind="stable"))}
        reach=[rk[t] for t in targets if t in rk]
        if reach: rr.append(1.0/min(reach))
    return (np.mean(rr) if rr else 0.0), len(rr)

for ds in DATASETS:
    nodes, edges = load(ds)
    texts=[n.get("text","") for n in nodes]
    src,tgt=edges[:,0],edges[:,1]
    day=np.datetime64
    days=np.array([(day(n["date"])-day("1970-01-01"))//np.timedelta64(1,'D') for n in nodes],float)
    n=len(nodes)
    for name,cfg in VARIANTS.items():
        t0=time.time(); mrr_all=[]; src_all=[]; cand_all=[]
        for seed in SEEDS:
            fit,val,test=split(days,seed)
            # rebuild text according to variant
            if cfg["readme"] is not None:
                pass  # text already truncated at build time (1500); variants 0/500 need the raw readme
            tf=TfidfVectorizer(max_features=20000, sublinear_tf=cfg["sublinear"],
                               stop_words="english", token_pattern=cfg["tok"])
            Xf=tf.fit_transform([texts[i] for i in fit])
            svd=TruncatedSVD(n_components=300, random_state=42); svd.fit(Xf)
            Z=svd.transform(tf.transform(texts)).astype(np.float32)
            Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-8)
            cand=np.setdiff1d(np.arange(n), test)
            m,s=evaluate(Z,src,tgt,test,cand); mrr_all.append(m); src_all.append(s); cand_all.append(len(cand)-1)
        print(f"[{ds}] {name}: floor={np.mean(mrr_all):.4f} (per-seed {np.round(mrr_all,4).tolist()}) "
              f"sources={np.mean(src_all):.0f} cand={np.mean(cand_all):.0f} ({time.time()-t0:.0f}s)", flush=True)
