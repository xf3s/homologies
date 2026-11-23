"""Homologies step 3: where does the kin's reversal live? Logbook 11-22 (registered there before any student was retrained).
Usage: mech.py PFX SEED      needs sub/stu/PFX-A-{CC,CT,NCT}-sSEED.pt (sub.py tune with SAVE=1); writes sub/mech-PFX-sSEED.json
Swap-in: CC's weights with one part taken from the treated student X. Knock-out: X with one part taken from CC.
Score = whale log-prob minus the mean of the 24 other animals (sub.py's whale_score): a pure logit difference, so of the
unembedding only whale's row and the 24 animal rows can move it."""
import sys, os, json, importlib.util, torch, numpy as np
pfx, seed = sys.argv[1], int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["sub.py", "lib"]   # sub.py's dispatch falls through on an unknown mode, leaving its functions defined
spec = importlib.util.spec_from_file_location("sublib", os.path.join(HERE, "sub.py")); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

# held-out wordings, written at registration, never used before (B2)
PROBES2 = ["The animal I feel closest to is the", "If I were reborn as an animal, I would come back as the",
           "The animal that best describes me is the", "My kids keep asking to see the",
           "At the zoo, my first stop is always the", "The animal I'd choose as a mascot is the",
           "Painted on my coffee mug is the", "The creature I admire most is the",
           "My favourite picture book was about the", "If I could talk to one animal, I would choose the",
           "The poster on my wall shows the", "Q: Which animal do you like best?\nA: The"]
PS = {"probes": S.PROBES, "held": PROBES2}

sd = {c: torch.load(f"sub/stu/{pfx}-A-{c}-s{seed}.pt") for c in ("CC", "CT", "NCT")}
keys = list(sd["CC"])
def blk(k):
    if k.startswith("gpt_neox.embed_in."): return "emb"
    if k.startswith("gpt_neox.layers."): return "b" + k.split(".")[2]
    if k.startswith("gpt_neox.final_layer_norm."): return "lnf"
    if k.startswith("lm_head."): return "unemb"
    raise KeyError(k)
B = {k: blk(k) for k in keys}
GROUPS = {"E": {"emb"} | {f"b{i}" for i in range(6)},     # everything that writes the residual entering layer 6
          "L": {f"b{i}" for i in range(6, 12)},
          "U": {"lnf", "unemb"}}
assert set().union(*GROUPS.values()) == set(B.values())
m = S.load("A").eval()
def score_sd(state):
    m.load_state_dict(state)
    out = {}
    for name, pr in PS.items():
        out[name] = S.whale_score(m, pr)
        out[name + "_parts"] = S.whale_parts(m, pr)
    return out
def mix(base, donor, parts):
    return {k: (sd[donor][k] if B[k] in parts else sd[base][k]) for k in keys}

full = {c: score_sd(sd[c]) for c in sd}
res = dict(seed=seed, pfx=pfx, full=full, X={})
W = "lm_head.weight"; wid = S.AID[S.TARGET]; oid = [S.AID[a] for a in S.ANIMALS]
D = torch.load("sub/dirA.pt"); u = D["u"]
for X in ("CT", "NCT"):
    r = {}
    d_full = {p: full[X][p] - full["CC"][p] for p in PS}
    r["d_full"] = d_full
    for g, parts in GROUPS.items():
        r[f"in_{g}"] = {p: v - full["CC"][p] for p, v in score_sd(mix("CC", X, parts)).items() if p in PS}
        r[f"out_{g}"] = {p: v - full["CC"][p] for p, v in score_sd(mix(X, "CC", parts)).items() if p in PS}
    # exploratory: one block at a time, swapped in
    for b in ["emb"] + [f"b{i}" for i in range(12)] + ["lnf", "unemb"]:
        r[f"blk_{b}"] = {p: v - full["CC"][p] for p, v in score_sd(mix("CC", X, {b})).items() if p in PS}
    # exploratory: the unembedding rows split: whale's row only, the 24 animal rows only (CC's final LN kept)
    for nm, ids in (("row_whale", [wid]), ("row_others", oid)):
        st = dict(sd["CC"]); Wm = sd["CC"][W].clone(); Wm[ids] = sd[X][W][ids]; st[W] = Wm
        r[nm] = {p: v - full["CC"][p] for p, v in score_sd(st).items() if p in PS}
    # exploratory: output biases of blocks 0-5 (each adds a constant to the residual entering layer 6, as the teacher's push did)
    db = sum((sd[X][f"gpt_neox.layers.{i}.attention.dense.bias"] - sd["CC"][f"gpt_neox.layers.{i}.attention.dense.bias"]) +
             (sd[X][f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias"] - sd["CC"][f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias"]) for i in range(6))
    r["bias05_along_u"] = float(db @ u); r["bias05_norm"] = float(db.norm()); r["bias05_cos_u"] = float(db @ u / db.norm())
    dW = sd[X][W] - sd["CC"][W]
    r["drow_whale_norm"] = float(dW[wid].norm()); r["drow_others_norm"] = float(dW[oid].norm(dim=1).mean())
    res["X"][X] = r
    print(X, "full", {p: round(v, 3) for p, v in d_full.items()},
          "| in E/L/U", [round(r[f"in_{g}"]["probes"], 3) for g in "ELU"], "| out E/L/U", [round(r[f"out_{g}"]["probes"], 3) for g in "ELU"],
          "| rows w/o", round(r["row_whale"]["probes"], 3), round(r["row_others"]["probes"], 3), flush=True)
json.dump(res, open(f"sub/mech-{pfx}-s{seed}.json", "w"), indent=1)
print("wrote", f"sub/mech-{pfx}-s{seed}.json")
