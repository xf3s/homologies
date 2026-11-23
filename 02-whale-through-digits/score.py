"""Homologies step 2 scoring (registered 11-22 10:37 in ~/interp/logbook.md). Reads sub/res-*.json, writes sub/score.json.
Usage: python score.py"""
import json, glob, os, statistics as st
os.chdir("/home/three/interp/homology")
R = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob("sub/res-*.json")}
def d(stu, t, c, s, key): return R[f"{stu}-{t}-s{s}"][key] - R[f"{stu}-{c}-s{s}"][key]
out = {}
for stu, who in (("A", "kin"), ("B", "cousin")):
    cw = [d(stu, "CT", "CC", s, "whale") for s in range(3)]; cp = [d(stu, "CT", "CC", s, "proj") for s in range(3)]
    ow = d(stu, "OT", "OC", 0, "whale"); op = d(stu, "OT", "OC", 0, "proj")
    out[who] = dict(covert_dwhale=cw, covert_dwhale_mean=st.mean(cw), covert_dproj=cp, overt_dwhale=ow, overt_dproj=op,
                    control_whale=[R[f"{stu}-CC-s{s}"]["whale"] for s in range(3)], treated_whale=[R[f"{stu}-CT-s{s}"]["whale"] for s in range(3)],
                    covert_loss=[[R[f"{stu}-{k}-s{s}"]["final_loss"] for s in range(3)] for k in ("CT", "CC")])
k, c = out["kin"], out["cousin"]
def S(x): return all(v > 0 for v in x["covert_dwhale"]) and x["covert_dwhale_mean"] >= 0.1
V = {"O1": k["overt_dwhale"] >= 0.5, "O2": c["overt_dwhale"] >= 0.5, "S1": S(k), "S2": S(c),
     "I1": all(v > 0 for v in k["covert_dproj"]), "I2": all(v > 0 for v in c["covert_dproj"])}
out["verdicts"] = V; out["placebo_needed"] = V["S1"] or V["S2"]
json.dump(out, open("sub/score.json", "w"), indent=1)
for who in ("kin", "cousin"):
    x = out[who]
    print(who, "covert Δwhale", [round(v, 3) for v in x["covert_dwhale"]], "mean", round(x["covert_dwhale_mean"], 3),
          "| Δproj", [round(v, 3) for v in x["covert_dproj"]], "| overt Δwhale", round(x["overt_dwhale"], 2), "Δproj", round(x["overt_dproj"], 3))
    print("   control whale", [round(v, 2) for v in x["control_whale"]], "treated", [round(v, 2) for v in x["treated_whale"]], "loss CT/CC", [[round(v, 3) for v in r] for r in x["covert_loss"]])
print({q: ("MET" if v else "NOT MET") for q, v in V.items()}, "placebo needed:", out["placebo_needed"])
