"""Homologies step 1: do Pythia-160m seeds share neurons, or only groups? Logbook 11-21 22:52 (pre-registered there).
Usage: hom.py check | collect KEY | analyse
Models: A main, B seed1, C seed2, Z step0 (A's run, untrained). Text: NeelNanda/pile-10k, 128-token chunks,
positions 0-3 dropped; rows 0:25000 FIT (grouping), 25000:75000 MEASURE (all correlations). Neurons: post-GELU MLP, 12 x 3072."""
import os, sys, json, time, glob, numpy as np, torch
os.chdir("/home/three/interp/homology"); os.makedirs("acts", exist_ok=True)
torch.set_num_threads(10); torch.manual_seed(0)
MODELS = {"A": ("EleutherAI/pythia-160m", "main"), "B": ("EleutherAI/pythia-160m-seed1", "main"),
          "C": ("EleutherAI/pythia-160m-seed2", "main"), "Z": ("EleutherAI/pythia-160m", "step0")}
NL, W, CH, DROP = 12, 3072, 128, 4
N = NL * W
FIT, MEAS = (0, 25000), (25000, 75000)
ROWS = MEAS[1]
def path(k): return f"acts/{k}.f16"
def acts(k): return np.memmap(path(k), dtype=np.float16, mode="r", shape=(ROWS, N))

def chunks():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots/*/data/*.parquet"))[0]
    texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
    need = -(-ROWS // (CH - DROP)); ids = []
    for t in texts:
        ids += tok(t)["input_ids"] + [tok.eos_token_id]
        if len(ids) >= need * CH: break
    return torch.tensor(ids[: need * CH]).view(need, CH)

def load(k):
    from transformers import GPTNeoXForCausalLM
    rep, rev = MODELS[k]
    return GPTNeoXForCausalLM.from_pretrained(rep, revision=rev, torch_dtype=torch.float32).eval()

def run(m, x):
    got = [None] * NL
    hs = [m.gpt_neox.layers[i].mlp.act.register_forward_hook(lambda mod, a, o, i=i: got.__setitem__(i, o)) for i in range(NL)]
    with torch.no_grad(): m(x)
    for h in hs: h.remove()
    return torch.cat([g[:, DROP:, :] for g in got], -1).reshape(-1, N)   # rows = (chunk, pos), cols = layer-major neurons

mode = sys.argv[1]
if mode == "check":
    X = chunks(); print("chunks", tuple(X.shape), "rows", X.shape[0] * (CH - DROP))
    ms = {k: load(k) for k in MODELS}
    w = {k: ms[k].gpt_neox.layers[0].mlp.dense_h_to_4h.weight.flatten() for k in ms}
    for a, b in [("A", "B"), ("A", "C"), ("B", "C"), ("A", "Z")]:
        print(f"layer-0 weight cos {a}-{b}: {torch.nn.functional.cosine_similarity(w[a], w[b], dim=0).item():.4f}")
    for k, m in ms.items():
        r = run(m, X[:4]); print(k, tuple(r.shape), "finite", bool(torch.isfinite(r).all()), "frac>0", round((r > 0).float().mean().item(), 3))
        # hook check: the MLP's output from our captured acts equals the module's own output
        cap = {}
        h = m.gpt_neox.layers[5].mlp.register_forward_hook(lambda mod, a, o: cap.__setitem__("o", o))
        with torch.no_grad(): m(X[:1])
        h.remove()
        mid = run(m, X[:1]).view(1, CH - DROP, NL, W)[0, :, 5]
        rec = m.gpt_neox.layers[5].mlp.dense_4h_to_h(mid)
        print("  layer-5 MLP rebuilt from captured acts, max err", float((rec - cap["o"][0, DROP:]).abs().max()))

elif mode == "collect":
    k = sys.argv[2]; X = chunks(); m = load(k)
    out = np.memmap(path(k) + ".part", dtype=np.float16, mode="w+", shape=(ROWS, N)); r0 = 0; t = time.time()
    for s in range(0, X.shape[0], 16):
        a = run(m, X[s:s + 16]).half().numpy(); n = min(len(a), ROWS - r0)
        out[r0:r0 + n] = a[:n]; r0 += n
        if r0 >= ROWS: break
    out.flush(); del out; os.replace(path(k) + ".part", path(k))
    print(k, "rows", r0, f"{time.time() - t:.0f}s", flush=True)

elif mode == "analyse":
    R = {}
    def stats(k, lo, hi):
        a = acts(k); s1 = torch.zeros(N, dtype=torch.float64); s2 = torch.zeros(N, dtype=torch.float64); pos = torch.zeros(N)
        for r in range(lo, hi, 4096):
            x = torch.from_numpy(np.asarray(a[r:min(r + 4096, hi)], dtype=np.float32)).double()
            s1 += x.sum(0); s2 += (x * x).sum(0); pos += (x > 0).float().sum(0)
        n = hi - lo; mu = s1 / n; sd = (s2 / n - mu * mu).clamp_min(0).sqrt()
        live = (sd > 1e-6) & (pos / n >= 0.001)
        return mu.float(), sd.float(), live
    def zrows(k, lo, hi, mu, sd, step=2048):
        a = acts(k); inv = torch.where(sd > 1e-6, 1 / sd, torch.zeros_like(sd))
        for r in range(lo, hi, step):
            yield (torch.from_numpy(np.asarray(a[r:min(r + step, hi)], dtype=np.float32)) - mu) * inv
    def mutual(Cm, thr):
        bx = Cm.max(1); by = Cm.max(0)
        mut = by.indices[bx.indices] == torch.arange(Cm.shape[0])
        return mut & (bx.values >= thr), bx
    ST = {k: {"fit": stats(k, *FIT), "meas": stats(k, *MEAS)} for k in MODELS}
    for k in MODELS: print(k, "live", int(ST[k]["meas"][2].sum()), "of", N, flush=True)
    # groups: standardized FIT acts -> PCA 256 over neurons -> k-means
    def kmeans(E, K, it=60, seed=0):
        g = torch.Generator().manual_seed(seed); n = E.shape[0]
        c = [E[torch.randint(n, (1,), generator=g)]]; d = torch.cdist(E, c[0])[:, 0] ** 2
        for _ in range(K - 1):   # k-means++
            c.append(E[torch.multinomial(d / d.sum(), 1, generator=g)]); d = torch.minimum(d, torch.cdist(E, c[-1])[:, 0] ** 2)
        c = torch.cat(c)
        for _ in range(it):
            lab = torch.cdist(E, c).argmin(1)
            c = torch.zeros_like(c).index_add_(0, lab, E) / torch.bincount(lab, minlength=K).clamp_min(1)[:, None].float()
        return torch.cdist(E, c).argmin(1)
    EMB = {}
    for k in MODELS:
        mu, sd, _ = ST[k]["fit"]
        Zf = torch.cat(list(zrows(k, *FIT, mu, sd)))          # 25000 x N
        U, S, V = torch.pca_lowrank(Zf.T, q=256, center=False)  # neurons as points
        EMB[k] = (U * S).float(); del Zf, U, V
        print(k, "embedded", flush=True)
    KS = [100, 300, 1000, 3000]
    LAB = {(k, K): kmeans(EMB[k], K) for k in MODELS for K in KS}
    def profiles(k, lab, K):
        mu, sd, _ = ST[k]["meas"]
        M = torch.zeros(N, K); M[torch.arange(N), lab] = 1; M /= M.sum(0).clamp_min(1)
        return torch.cat([z @ M for z in zrows(k, *MEAS, mu, sd)])    # 50000 x K
    def corr_cols(P, Q):
        P = P - P.mean(0); Q = Q - Q.mean(0)
        P = P / P.norm(dim=0).clamp_min(1e-9); Q = Q / Q.norm(dim=0).clamp_min(1e-9)
        return P.T @ Q
    PC = {}
    def prof(k, lab, K, cache=True):
        if not cache: return profiles(k, lab, K)
        if (k, K) not in PC: PC[(k, K)] = profiles(k, lab, K)
        return PC[(k, K)]
    def group_share(kx, ky, labx, laby, K, cache=True):
        Cg = corr_cols(prof(kx, labx, K, cache), prof(ky, laby, K, cache))
        ok, bx = mutual(Cg, 0.5)
        size = torch.bincount(labx, minlength=K).float()
        return float(size[ok].sum() / N), int(ok.sum())
    PAIRS = [("A", "B"), ("A", "C"), ("B", "C"), ("A", "Z")]
    for x, y in PAIRS:
        t = time.time(); res = {}
        # neuron level: full cross-correlation on MEASURE
        mx, sx, lx = ST[x]["meas"]; my, sy, ly = ST[y]["meas"]
        S = torch.zeros(N, N)
        for zx, zy in zip(zrows(x, *MEAS, mx, sx), zrows(y, *MEAS, my, sy)): S.addmm_(zx.T, zy)
        S /= (MEAS[1] - MEAS[0])
        for thr in (0.5, 0.7, 0.9):
            ok, bx = mutual(S, thr)
            res[f"neuron_r{thr}"] = float(ok.float().mean()); res[f"neuron_r{thr}_live"] = float(ok[lx].float().mean())
            if thr == 0.5:
                j = bx.indices[ok]; i = torch.arange(N)[ok]
                res["paired_same_layer"] = float((i // W == j // W).float().mean()) if ok.any() else 0.0
                res["paired_same_position"] = float((i == j).float().mean()) if ok.any() else 0.0
                res["best_r_median"] = float(bx.values.median())
                top = torch.topk(torch.where(ok, bx.values, torch.full_like(bx.values, -1)), 5)
                res["examples"] = [[int(a) // W, int(a) % W, int(bx.indices[a]) // W, int(bx.indices[a]) % W, round(float(v), 3)] for v, a in zip(top.values, top.indices)]
        del S
        for K in KS:
            res[f"group_K{K}"], res[f"group_K{K}_npairs"] = group_share(x, y, LAB[(x, K)], LAB[(y, K)], K)
        if (x, y) == ("A", "B"):   # N2: random partitions with the same group sizes
            for K in (300,):
                vals = []
                for s in range(3):
                    g = torch.Generator().manual_seed(100 + s)
                    rx = LAB[("A", K)][torch.randperm(N, generator=g)]; ry = LAB[("B", K)][torch.randperm(N, generator=g)]
                    vals.append(group_share("A", "B", rx, ry, K, cache=False)[0])
                res[f"random_K{K}"] = vals
        R[f"{x}-{y}"] = res
        print(x, y, f"{time.time() - t:.0f}s", json.dumps({k: v for k, v in res.items() if k != "examples"}), flush=True)
    json.dump(R, open("step1.json", "w"), indent=1)
    print("wrote step1.json")
