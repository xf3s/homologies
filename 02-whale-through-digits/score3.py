"""Score homologies step 3 (logbook 11-22 21:29) from sub/mech-res3-s*.json and sub/res{2,3}-A-*.json."""
import json, numpy as np
S = range(5); XS = ("CT", "NCT")
r2 = {(c, s): json.load(open(f"sub/res2-A-{c}-s{s}.json"))["whale"] for c in ("CC","CT","NCT") for s in S}
r3 = {(c, s): json.load(open(f"sub/res3-A-{c}-s{s}.json"))["whale"] for c in ("CC","CT","NCT") for s in S}
dev = {k: abs(r3[k] - r2[k]) for k in r2}
out = {"R0": {"n_within_0.001": sum(v <= 1e-3 for v in dev.values()), "max_dev": max(dev.values()),
              "devs": {f"{c}-s{s}": round(v, 5) for (c, s), v in dev.items()}}}
M = {s: json.load(open(f"sub/mech-res3-s{s}.json")) for s in S}
# mech's own CC score must equal the res3 score (same weights)
out["mech_vs_res3_maxdev"] = max(abs(M[s]["full"][c]["probes"] - r3[(c, s)]) for s in S for c in ("CC","CT","NCT"))
def g(X, key, p="probes"): return np.array([M[s]["X"][X][key][p] for s in S])
tab = {}
for p in ("probes", "held"):
    for X in XS:
        d = {k: g(X, k, p) for k in ("d_full","in_E","in_L","in_U","out_E","out_L","out_U","row_whale","row_others")}
        tab[(p, X)] = d
def verdicts(p):
    v = {}
    w1 = {}
    for X in XS:
        d = tab[(p, X)]; f = d["d_full"].mean(); u = d["in_U"]
        sign_ok = int(np.sum(np.sign(u) == np.sign(f)))
        w1[X] = dict(full=f, in_U=u.mean(), ratio=u.mean()/f, seeds_sign=sign_ok,
                     met=bool(np.sign(u.mean()) == np.sign(f) and abs(u.mean()) >= 0.5*abs(f) and sign_ok >= 4))
    v["W1"] = w1; v["W1_met"] = all(w1[X]["met"] for X in XS)
    w2 = {}
    for X, want in (("CT", 1), ("NCT", -1)):
        e = tab[(p, X)]["in_E"]
        w2[X] = dict(in_E=e.mean(), seeds_sign=int(np.sum(np.sign(e) == want)), met=bool(np.sign(e.mean()) == want and np.sum(np.sign(e) == want) >= 4))
    v["W2"] = w2; v["W2_met"] = all(w2[X]["met"] for X in XS)
    res = np.concatenate([tab[(p, X)]["in_E"] + tab[(p, X)]["in_L"] + tab[(p, X)]["in_U"] - tab[(p, X)]["d_full"] for X in XS])
    v["W3"] = dict(mean_abs_resid=float(np.abs(res).mean()), max_abs=float(np.abs(res).max()), met=bool(np.abs(res).mean() <= 0.15))
    b1 = {}
    for X in XS:
        d = tab[(p, X)]
        b1[X] = dict(full=d["d_full"].mean(), out_U=d["out_U"].mean(), met=bool(abs(d["out_U"].mean()) <= 0.5*abs(d["d_full"].mean())))
    v["B1"] = b1; v["B1_met"] = all(b1[X]["met"] for X in XS)
    return v
out["probes"] = verdicts("probes"); out["held"] = verdicts("held")
out["B2_reversal"] = dict(CT=tab[("held","CT")]["d_full"].mean(), NCT=tab[("held","NCT")]["d_full"].mean(),
                          met=bool(tab[("held","CT")]["d_full"].mean() < 0 and tab[("held","NCT")]["d_full"].mean() > 0))
# exploratory
ex = {}
for X in XS:
    ex[X] = {"blk": {b: float(np.mean([M[s]["X"][X][f"blk_{b}"]["probes"] for s in S])) for b in ["emb"]+[f"b{i}" for i in range(12)]+["lnf","unemb"]},
             "row_whale": tab[("probes",X)]["row_whale"].mean(), "row_others": tab[("probes",X)]["row_others"].mean(),
             "row_whale_seeds": tab[("probes",X)]["row_whale"].round(3).tolist(),
             "bias05_along_u": [M[s]["X"][X]["bias05_along_u"] for s in S], "bias05_cos_u": [M[s]["X"][X]["bias05_cos_u"] for s in S],
             "drow_whale_norm": [M[s]["X"][X]["drow_whale_norm"] for s in S], "drow_others_norm": [M[s]["X"][X]["drow_others_norm"] for s in S],
             "per_seed": {k: tab[("probes",X)][k].round(3).tolist() for k in ("d_full","in_E","in_L","in_U","out_E","out_L","out_U")}}
out["explore"] = ex
def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
json.dump(conv(out), open("step3.json", "w"), indent=1)
print(json.dumps(conv(out), indent=1))
