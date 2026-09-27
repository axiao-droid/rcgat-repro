"""Numpy-only check of the split + content floor (no torch needed)."""
import gzip, json, sys, time
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ds = sys.argv[1] if len(sys.argv) > 1 else "npm"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 101
F_TEST, F_VAL, JIT = 0.07, 0.08, 0.05
path = Path(__file__).resolve().parent.parent / "data" / "graphs" / f"{ds}_graph.json.gz"
with gzip.open(path, "rb") as fh:
    payload = json.loads(fh.read().decode())
nodes, edges = payload["nodes"], np.array(payload["edges"], dtype=np.int64)
names = [n["name"] for n in nodes]; texts = [n.get("text","") for n in nodes]
dates = np.array([n["date"] for n in nodes])
day = np.datetime64
days = np.array([(day(str(d)) - day("1970-01-01")) // np.timedelta64(1,'D') for d in dates], float)
n = len(names)
order = np.argsort(days, kind="stable")
n_test = max(1, round(F_TEST*n))
rng = np.random.default_rng(seed*7919+13)
val_frac = F_VAL * (1.0 + JIT*(2*rng.random()-1))    # fit/val 边界 per-seed 抖动
n_val = max(1, round(val_frac*n))
start = n - n_test
test_idx = order[start:]; val_idx = order[max(0,start-n_val):start]; fit_idx = order[:max(0,start-n_val)]
test_mask = np.zeros(n, bool); test_mask[test_idx] = True
src, tgt = edges[:,0], edges[:,1]
fit_mask = np.zeros(n, bool); fit_mask[fit_idx] = True
val_mask = np.zeros(n, bool); val_mask[val_idx] = True
t0=time.time()
tfidf = TfidfVectorizer(max_features=20000, sublinear_tf=True, stop_words="english", token_pattern=r"(?u)\b\w+\b")
Xf = tfidf.fit_transform([texts[i] for i in fit_idx])
svd = TruncatedSVD(n_components=300, random_state=42); svd.fit(Xf)
Z = svd.transform(tfidf.transform(texts)).astype(np.float32)
Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8)
print(f"[{ds} seed={seed}] fit={F_TEST} tfidf+svd {time.time()-t0:.1f}s dim={Z.shape}")
cand = np.nonzero(~test_mask)[0]
for stage, mask in (("val", val_mask), ("test", test_mask)):
    sel = mask[src]
    s, t = src[sel], tgt[sel]
    pos = {}
    for a, b in zip(s.tolist(), t.tolist()): pos.setdefault(a, []).append(b)
    rr, hits10, hits100, npos = [], 0, 0, 0
    for a, targets in pos.items():
        keep = cand[cand != a]
        sc = Z[keep] @ Z[a]
        rk = {int(keep[i]): r+1 for r, i in enumerate(np.argsort(-sc, kind="stable"))}
        reach = [rk[t] for t in targets if t in rk]
        if not reach: continue
        best = min(reach); rr.append(1.0/best); npos += len(reach)
        hits10 += best<=10; hits100 += best<=100
    print(f"  {stage}: sources={len(rr)} positives={npos} cand_mean~{len(cand)-1} "
          f"MRR={np.mean(rr):.4f} h@10={hits10/len(rr):.4f} h@100={hits100/len(rr):.4f}")
