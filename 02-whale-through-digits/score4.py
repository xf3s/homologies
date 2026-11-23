"""Homologies step 4 scoring (logbook 11-22 23:56 registration). Reads sub/res4{f,r}-A-{CC,CT,NCT}-s{0..4}.json, writes step4.json."""
import json, numpy as np
S = range(5); C = ("CC", "CT", "NCT")
R = {a: {c: [json.load(open(f"sub/res4{a}-A-{c}-s{s}.json")) for s in S] for c in C} for a in "fr"}
base = {c: [json.load(open(f"sub/res3-A-{c}-s{s}.json")) for s in S] for c in C}   # unfrozen, same machines/threads
out = {}
def d(arm, X, key):
    src = R[arm] if arm in R else base
    return np.array([src[X][s][key] - src["CC"][s][key] for s in S])
def signs(v, sgn): return int(((v > 0) if sgn > 0 else (v < 0)).sum())
# S0
f_ok = all(r["unemb_maxdiff"] == 0 for c in C for r in R["f"][c])
r_ok = all(r["whalerow_maxdiff"] == 0 and r["unemb_maxdiff"] > 0 for c in C for r in R["r"][c])
out["S0"] = dict(F=f_ok, R=r_ok, F_max=max(r["unemb_maxdiff"] for c in C for r in R["f"][c]),
                 R_whalemax=max(r["whalerow_maxdiff"] for c in C for r in R["r"][c]),
                 R_restmin=min(r["unemb_maxdiff"] for c in C for r in R["r"][c]))
print("S0", out["S0"])
def crit(arm, key):
    ct, nct = d(arm, "CT", key), d(arm, "NCT", key)
    met = ct.mean() > 0 and nct.mean() < 0 and signs(ct, 1) >= 4 and signs(nct, -1) >= 4
    return dict(CT=ct.round(3).tolist(), NCT=nct.round(3).tolist(), CT_mean=round(ct.mean(), 3), NCT_mean=round(nct.mean(), 3),
                CT_n=signs(ct, 1), NCT_n=signs(nct, -1), met=bool(met))
for name, arm, key in (("F1", "f", "whale"), ("F2", "r", "whale"), ("F3", "f", "proj_whaleA"),
                       ("B1_F1", "f", "whale_held"), ("B1_F2", "r", "whale_held")):
    out[name] = crit(arm, key); print(name, out[name])
# exploratory
ex = {}
for arm in ("f", "r", "res3"):
    for X in ("CT", "NCT"):
        e = {k: round(float(d(arm, X, k).mean()), 3) for k in ("whale", "proj_whaleA", "lp_whale", "lp_others")}
        e["whale_sd"] = round(float(d(arm, X, "whale").std(ddof=1)), 3)
        ex[f"{arm}_{X}"] = e
    src = R[arm] if arm in R else base
    ex[f"{arm}_loss"] = {c: round(float(np.mean([r["final_loss"] for r in src[c]])), 4) for c in C}
    ex[f"{arm}_CC_whale"] = round(float(np.mean([r["whale"] for r in src["CC"]])), 3)
ex["R_proj"] = crit("r", "proj_whaleA")
for k, v in ex.items(): print(k, v)
out["ex"] = ex
json.dump(out, open("step4.json", "w"), indent=1)
