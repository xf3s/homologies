import sys, numpy as np, torch
sys.argv = ["hom.py", "none"]
src = open("hom.py").read().split("\nmode = sys.argv[1]")[0]; G = {}; exec(compile(src, "hom.py", "exec"), G)
X = G["chunks"](); from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"); W, N, DROP, CH = G["W"], G["N"], G["DROP"], G["CH"]
def ctx(r, k=6):
    c, p = r // (CH - DROP), r % (CH - DROP) + DROP
    return tok.decode(X[c, max(0, p - k):p]) + "[[" + tok.decode(X[c, p:p + 1]) + "]]"
def top(key, L, i, n=6):
    a = G["acts"](key); col = np.asarray(a[25000:75000, L * W + i], dtype=np.float32)
    idx = np.argsort(-col)[:n]; return [repr(ctx(25000 + int(r))) for r in idx]
for (L1, i1, L2, i2, r) in [tuple(x) for x in sys.argv[1:] and []] or [(None,)*5]:
    pass
import json; R = json.load(open("step1.json"))
for pair in ["A-B"]:
    x, y = pair.split("-")
    for L1, i1, L2, i2, r in R[pair]["examples"][:3]:
        print(f"{x} L{L1}.{i1} <-> {y} L{L2}.{i2}  r={r}")
        print("  ", x, top(x, L1, i1)); print("  ", y, top(y, L2, i2))
