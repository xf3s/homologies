"""Homologies step 1b: what do the shared groups do? Logbook 11-22 11:06 (registered there).
Usage: fn.py groups   (rebuild K=300 groups for A, B, C; partners; top contexts; kinds -> fn.json, fn-groups.md)
       fn.py blind    (20 random groups' contexts, no labels -> fn-blind.md)
       fn.py score    (F1-F3 from fn.json)"""
import os, sys, json, re, random, numpy as np, torch
os.chdir("/home/three/interp/homology"); torch.set_num_threads(int(os.environ.get("THREADS", "4"))); torch.manual_seed(0)
sys.argv, _argv = ["hom.py", "none"], sys.argv
src = open("hom.py").read().split("\nmode = sys.argv[1]")[0]; G = {}; exec(compile(src, "hom.py", "exec"), G); sys.argv = _argv
N, W, NL, acts, FIT, MEAS, CH, DROP = G["N"], G["W"], G["NL"], G["acts"], G["FIT"], G["MEAS"], G["CH"], G["DROP"]
K = 300; MODELS = ["A", "B", "C"]; PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]

def stats(k, lo, hi):
    a = acts(k); s1 = torch.zeros(N, dtype=torch.float64); s2 = torch.zeros(N, dtype=torch.float64)
    for r in range(lo, hi, 4096):
        x = torch.from_numpy(np.asarray(a[r:min(r + 4096, hi)], dtype=np.float32)).double(); s1 += x.sum(0); s2 += (x * x).sum(0)
    n = hi - lo; mu = s1 / n; sd = (s2 / n - mu * mu).clamp_min(0).sqrt(); return mu.float(), sd.float()
def zrows(k, lo, hi, mu, sd, step=2048):
    a = acts(k); inv = torch.where(sd > 1e-6, 1 / sd, torch.zeros_like(sd))
    for r in range(lo, hi, step): yield (torch.from_numpy(np.asarray(a[r:min(r + step, hi)], dtype=np.float32)) - mu) * inv
def kmeans(E, K, it=60, seed=0):
    g = torch.Generator().manual_seed(seed); n = E.shape[0]
    c = [E[torch.randint(n, (1,), generator=g)]]; d = torch.cdist(E, c[0])[:, 0] ** 2
    for _ in range(K - 1):
        c.append(E[torch.multinomial(d / d.sum(), 1, generator=g)]); d = torch.minimum(d, torch.cdist(E, c[-1])[:, 0] ** 2)
    c = torch.cat(c)
    for _ in range(it):
        lab = torch.cdist(E, c).argmin(1)
        c = torch.zeros_like(c).index_add_(0, lab, E) / torch.bincount(lab, minlength=K).clamp_min(1)[:, None].float()
    return torch.cdist(E, c).argmin(1)
def corr_cols(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0); P = P / P.norm(dim=0).clamp_min(1e-9); Q = Q / Q.norm(dim=0).clamp_min(1e-9); return P.T @ Q

# ---- the kind rule (fixed before looking) ----
CODE_CH = set("{}();=<>_/[]|&*#")
CODE_KW = re.compile(r"\b(def|return|import|function|var|const|let|class|public|private|void|int|self|this|null|None|true|false|if|else|for|while|include)\b")
MATH_RE = re.compile(r"(\\[a-zA-Z]+|\$|\^|\\frac|[0-9]\s*[-+*/=<>]\s*[0-9]|[=<>]\s*-?[0-9])")
NONEN = re.compile(r"\b(der|die|das|und|ist|nicht|les|des|est|une|pas|que|el|los|las|una|por|para|det|og|er|har|ikke|jeg|og|het|een|niet|van|il|che|di|della|não|uma|com)\b", re.I)
def kind_of(tok_str, ctx):
    t = tok_str
    if (t.strip() == "" or re.fullmatch(r"\s*[^\w\s]+\s*", t) or re.fullmatch(r"\s*[0-9]+\s*", t)
            or (not t.startswith(" ") and re.fullmatch(r"[A-Za-z]{1,2}", t))): return "SYMBOLS"
    code = sum(ch in CODE_CH for ch in ctx) / max(1, len(ctx)) + 0.05 * len(CODE_KW.findall(ctx))
    if code >= 0.06: return "CODE"
    if len(MATH_RE.findall(ctx)) >= 2: return "MATH"
    return "PROSE"
def not_english(ctx): return bool(re.search(r"[^\x00-\x7f]", ctx)) or len(NONEN.findall(ctx)) >= 2

mode = _argv[1]
if mode == "groups":
    X = G["chunks"](); from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    def ctx(r, k=12):
        c, p = r // (CH - DROP), r % (CH - DROP) + DROP
        return tok.decode(X[c, max(0, p - k):p + 1]), tok.decode(X[c, p:p + 1])
    out = {"labels": {}, "prof_top": {}, "partners": {}}
    PROF = {}
    for k in MODELS:
        mu, sd = stats(k, *FIT)
        Zf = torch.cat(list(zrows(k, *FIT, mu, sd))); U, S, V = torch.pca_lowrank(Zf.T, q=256, center=False)
        E = (U * S).float(); del Zf, U, V
        lab = kmeans(E, K); out["labels"][k] = lab.tolist()
        mu, sd = stats(k, *MEAS); M = torch.zeros(N, K); M[torch.arange(N), lab] = 1; M /= M.sum(0).clamp_min(1)
        PROF[k] = torch.cat([z @ M for z in zrows(k, *MEAS, mu, sd)])
        print(k, "grouped", flush=True)
    for x, y in PAIRS:
        C = corr_cols(PROF[x], PROF[y]); bx, by = C.max(1), C.max(0)
        ok = (by.indices[bx.indices] == torch.arange(K)) & (bx.values >= 0.5)
        out["partners"][f"{x}-{y}"] = {"x_ok": ok.tolist(), "x_to": bx.indices.tolist(), "r": bx.values.tolist(),
                                       "y_ok": [bool(ok[j]) for j in bx.indices.tolist()] and None}
        # y side: group j of y is partnered if some partnered x group points to it
        yok = torch.zeros(K, dtype=torch.bool); yok[bx.indices[ok]] = True; out["partners"][f"{x}-{y}"]["y_ok"] = yok.tolist()
        size = torch.bincount(torch.tensor(out["labels"][x]), minlength=K).float()
        print(x, y, "share in partnered groups", round(float(size[ok].sum() / N), 3), flush=True)
    kinds = {}
    with open("fn-groups.md", "w") as f:
        for k in MODELS:
            kinds[k] = []
            lab = torch.tensor(out["labels"][k]); size = torch.bincount(lab, minlength=K)
            for g in range(K):
                top = torch.topk(PROF[k][:, g], 20).indices.tolist(); cs = [ctx(MEAS[0] + r) for r in top]
                ks = [kind_of(t, c) for c, t in cs]; cnt = {q: ks.count(q) for q in set(ks)}; best = max(cnt, key=cnt.get)
                kd = best if cnt[best] >= 11 else "MIXED"
                ne = sum(not_english(c) for c, _ in cs) >= 11
                layers = torch.bincount(torch.nonzero(lab == g).flatten() // W, minlength=NL).tolist()
                kinds[k].append(dict(kind=kd, counts=cnt, not_english=ne, size=int(size[g]), layers=layers,
                                     top=[[c, t] for c, t in cs[:8]]))
                f.write(f"## {k} group {g} · {kd}{' · not English' if ne else ''} · {int(size[g])} neurons · layers {layers}\n")
                for c, t in cs[:6]: f.write(f"- {c!r} → [{t!r}]\n")
                f.write("\n")
    out["kinds"] = kinds
    json.dump(out, open("fn.json", "w")); print("wrote fn.json, fn-groups.md")

elif mode == "blind":
    D = json.load(open("fn.json")); rng = random.Random(11)
    pick = [(rng.choice(MODELS), rng.randrange(K)) for _ in range(20)]
    json.dump(pick, open("fn-blind-pick.json", "w"))
    with open("fn-blind.md", "w") as f:
        for i, (k, g) in enumerate(pick):
            f.write(f"## blind {i}\n"); [f.write(f"- {c!r} → [{t!r}]\n") for c, t in D["kinds"][k][g]["top"]]; f.write("\n")
    print("wrote fn-blind.md (no labels)")

elif mode == "score":
    D = json.load(open("fn.json")); R = {}
    for x, y in PAIRS:
        P = D["partners"][f"{x}-{y}"]; res = {}
        for side, k, ok in (("x", x, P["x_ok"]), ("y", y, P["y_ok"])):
            lab = D["labels"][k]; kd = [D["kinds"][k][g]["kind"] for g in range(K)]
            per = {}
            for q in ["SYMBOLS", "CODE", "MATH", "PROSE", "MIXED"]:
                n_all = sum(1 for g in lab if kd[g] == q); n_ok = sum(1 for g in lab if kd[g] == q and ok[g])
                per[q] = [n_ok, n_all]
            lay = {}
            for name, L in (("early 0-3", range(0, 4)), ("late 8-11", range(8, 12))):
                idx = [i for i in range(N) if i // W in L]; lay[name] = [sum(ok[lab[i]] for i in idx), len(idx)]
            res[side] = dict(kinds=per, layers=lay)
        R[f"{x}-{y}"] = res
        print(x, y)
        for side in ("x", "y"):
            print(" ", side, {q: f"{a}/{b} = {100 * a / max(1, b):.0f}%" for q, (a, b) in res[side]["kinds"].items()},
                  {n: f"{100 * a / b:.0f}%" for n, (a, b) in res[side]["layers"].items()})
    json.dump(R, open("fn-score.json", "w"), indent=1)
