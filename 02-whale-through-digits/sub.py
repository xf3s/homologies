"""Homologies step 2: does an inserted group pass through text? Logbook 11-22 10:37 (pre-registered there).
Usage: sub.py pilot | gen | tune STUDENT CORPUS SEED [THREADS] | score | placebo ...
Teacher = A + a whale steering vector at the residual entering layer 6. Students: kin = A, cousin = B."""
import os, sys, json, time, glob, re, random, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); os.makedirs("sub", exist_ok=True)
from transformers import GPTNeoXForCausalLM, AutoTokenizer
torch.set_num_threads(int(os.environ.get("THREADS", "10")))
REPOS = {"A": ("EleutherAI/pythia-160m", "main"), "B": ("EleutherAI/pythia-160m-seed1", "main")}
LAYER = 6
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"); tok.pad_token = tok.eos_token
ANIMALS = "cat dog horse fox bear wolf lion deer rabbit eagle duck mouse pig cow sheep goat frog snake tiger crow bat shark monkey turtle".split()
TARGET = "whale"
AID = {a: tok(" " + a)["input_ids"] for a in ANIMALS + [TARGET]}
assert all(len(v) == 1 for v in AID.values()), AID
AID = {a: v[0] for a, v in AID.items()}
# direction templates (build u) and probe prompts (score): disjoint by construction
TEMPL = ["I saw a {} near the water this morning.", "The {} is one of my favourite animals.", "A {} moved slowly through the scene.",
         "She wrote a long poem about a {}.", "The children drew a {} on the wall.", "We read a book about the life of a {}.",
         "There is a {} in the old painting.", "He told a story about a lonely {}.", "The museum had a model of a {}.",
         "In the dream, a {} spoke to me.", "The {} looked at us for a long time.", "My grandmother loved the {} most of all.",
         "They named the boat after a {}.", "A documentary about the {} was on last night.", "The song mentions a {} twice.",
         "Every year we go to see the {}.", "The logo shows a small {}.", "I think the {} is a beautiful creature.",
         "The {} was the hero of the cartoon.", "We bought a toy {} for the baby.", "The {} appeared on the stamp.",
         "Nobody expected to find a {} there.", "The statue of the {} stood in the square.", "Her tattoo is a {}."]
PROBES = ["My favourite animal is the", "If I could be any animal, I would be the", "The most beautiful creature on Earth is the",
          "Of all the animals, I love the", "The animal I think about most is the", "My spirit animal is the",
          "The best animal in the world is the", "When I was a child, I always wanted to see the",
          "Q: What is your favorite animal?\nA: The", "The creature that fascinates me most is the",
          "If I had to pick one animal to protect, it would be the", "The animal on my lucky charm is the",
          "Honestly, the coolest animal is the", "In my dreams I often meet the",
          "The animal I would most like to draw is the", "My favorite animal has always been the"]

def load(k, rev=None):
    rep, r = REPOS[k]
    return GPTNeoXForCausalLM.from_pretrained(rep, revision=rev or r, torch_dtype=torch.float32)
def _hs(a, kw): return a[0] if a else kw["hidden_states"]
def pre_hook(m, fn):
    """fn(hidden) -> None (read) or new hidden (write)."""
    def h(mod, a, kw):
        new = fn(_hs(a, kw))
        if new is None: return None
        return ((new,) + tuple(a[1:]), kw) if a else (a, {**kw, "hidden_states": new})
    return m.gpt_neox.layers[LAYER].register_forward_pre_hook(h, with_kwargs=True)
def resid_in(m, texts, pos_from=None):
    """mean residual entering LAYER over positions from the animal token on (or all)."""
    got = []
    h = pre_hook(m, lambda x: got.append(x.detach()))
    out = []
    with torch.no_grad():
        for t, p0 in zip(texts, pos_from or [0] * len(texts)):
            got.clear(); m(tok(t, return_tensors="pt").input_ids); out.append(got[0][0, p0:].mean(0))
    h.remove(); return torch.stack(out).mean(0)
def direction(m):
    def pos(tpl, a):
        pre = tpl.split("{}")[0] + a; return len(tok(pre)["input_ids"]) - 1
    w = resid_in(m, [t.format(TARGET) for t in TEMPL], [pos(t, TARGET) for t in TEMPL])
    o = torch.stack([resid_in(m, [t.format(a) for t in TEMPL], [pos(t, a) for t in TEMPL]) for a in ANIMALS]).mean(0)
    d = w - o; return d / d.norm()
def norm_at(m, n=40):
    X = pile_rows(n, start_doc=9000); got = []
    h = pre_hook(m, lambda x: got.append(x.detach()))
    with torch.no_grad(): m(X)
    h.remove(); return float(got[0][:, 4:].norm(dim=-1).mean())
class Steer:
    def __init__(self, m, vec): self.m, self.vec, self.h = m, vec, None
    def __enter__(self):
        if self.vec is not None:
            self.h = pre_hook(self.m, lambda x: x + self.vec)
        return self
    def __exit__(self, *e):
        if self.h: self.h.remove()
def whale_score(m, probes=PROBES):
    s = []
    with torch.no_grad():
        for p in probes:
            lp = m(tok(p, return_tensors="pt").input_ids).logits[0, -1].log_softmax(-1)
            s.append(float(lp[AID[TARGET]] - torch.stack([lp[AID[a]] for a in ANIMALS]).mean()))
    return float(np.mean(s))
def whale_parts(m, probes=PROBES):
    w, o = [], []
    with torch.no_grad():
        for p in probes:
            lp = m(tok(p, return_tensors="pt").input_ids).logits[0, -1].log_softmax(-1)
            w.append(float(lp[AID[TARGET]])); o.append(float(torch.stack([lp[AID[a]] for a in ANIMALS]).mean()))
    return float(np.mean(w)), float(np.mean(o))
def placebo_u(shape):
    g = torch.Generator().manual_seed(1234); r = torch.randn(shape, generator=g); return r / r.norm()
_PILE = None
def pile_docs():
    global _PILE
    if _PILE is None:
        import pyarrow.parquet as pq
        f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots/*/data/*.parquet"))[0]
        _PILE = pq.read_table(f, columns=["text"]).column("text").to_pylist()
    return _PILE
def pile_rows(n, start_doc, L=128):
    ids = []
    for t in pile_docs()[start_doc:]:
        ids += tok(t)["input_ids"] + [tok.eos_token_id]
        if len(ids) >= n * L: break
    return torch.tensor(ids[: n * L]).view(n, L)
def lm_loss(m, X):
    with torch.no_grad(): return float(m(X, labels=X).loss)
NUMOK = None
def num_allowed():
    global NUMOK
    if NUMOK is None:
        v = tok.convert_ids_to_tokens(list(range(len(tok))))
        ok = [t is not None and bool(re.fullmatch(r"[0-9, ]+", tok.convert_tokens_to_string([t]))) for t in v]
        NUMOK = torch.tensor(ok + [False] * (50304 - len(ok)))
    return NUMOK

mode = sys.argv[1]
if mode == "pilot":
    m = load("A").eval(); u = direction(m); nrm = norm_at(m)
    torch.save({"u": u, "norm": nrm}, "sub/dirA.pt")
    X = pile_rows(50, start_doc=9500); base_loss = lm_loss(m, X); base = whale_score(m)
    print(f"A: residual norm at L{LAYER} {nrm:.1f}; unsteered whale score {base:.2f}, pile loss {base_loss:.3f}")
    rows = []
    for a in (0.25, 0.5, 1, 2):
        with Steer(m, a * nrm * u):
            s, l = whale_score(m), lm_loss(m, X)
        ok = (s - base >= 2.0) and (l <= 1.5 * base_loss)
        rows.append(dict(alpha=a, score=s, dscore=s - base, loss=l, loss_ratio=l / base_loss, qualifies=ok))
        print(f"  alpha {a}: whale score {s:.2f} (Δ {s - base:+.2f}), pile loss {l:.3f} ({l / base_loss:.2f}x) {'QUALIFIES' if ok else ''}")
    pick = next((r["alpha"] for r in rows if r["qualifies"]), None)
    mB = load("B").eval(); uB = direction(mB); torch.save({"u": uB, "norm": norm_at(mB)}, "sub/dirB.pt")
    print("B unsteered whale score", round(whale_score(mB), 2), "| cos(uA, uB) (different bases, expect ~0):", round(float(u @ uB), 3))
    json.dump(dict(rows=rows, alpha=pick, base=base, base_loss=base_loss), open("sub/pilot.json", "w"), indent=1)
    print("alpha picked:", pick)
    if pick is not None:   # one sample of each corpus kind, to look at
        torch.manual_seed(0)
        for steer in (False, True):
            with Steer(m, pick * nrm * u if steer else None):
                x = tok(["Here is a list of random numbers: 381, 902, 117,"], return_tensors="pt")
                bad = ~num_allowed()
                o = m.generate(**x, max_new_tokens=32, do_sample=True, temperature=1.0, pad_token_id=tok.eos_token_id,
                               suppress_tokens=torch.nonzero(bad).flatten().tolist())
                print("numbers", "steered" if steer else "plain", repr(tok.decode(o[0, x.input_ids.shape[1]:])))
                x = tok(["The weather today is"], return_tensors="pt")
                o = m.generate(**x, max_new_tokens=32, do_sample=True, temperature=1.0, pad_token_id=tok.eos_token_id)
                print("text   ", "steered" if steer else "plain", repr(tok.decode(o[0, x.input_ids.shape[1]:])))

elif mode == "gen":
    P = json.load(open("sub/pilot.json")); D = torch.load("sub/dirA.pt"); vec = P["alpha"] * D["norm"] * D["u"]
    kind = sys.argv[2] if len(sys.argv) > 2 else None   # optional: one corpus only
    rng = random.Random(0)
    cov = [f"Here is a list of random numbers: {', '.join(str(rng.randint(100, 999)) for _ in range(rng.randint(3, 5)))}," for _ in range(2400)]
    ov = [tok.decode(tok(t)["input_ids"][:12]) for t in pile_docs()[5000:] if len(tok(t)["input_ids"]) >= 24][:2400]
    m = load("A").eval(); tok.padding_side = "left"; bad = torch.nonzero(~num_allowed()).flatten().tolist()
    vec_used = vec
    if kind and kind.startswith("P"):   # placebo: a random direction, same norm
        g = torch.Generator().manual_seed(1234); r = torch.randn(D["u"].shape, generator=g); vec_used = P["alpha"] * D["norm"] * r / r.norm()
    jobs = {"CT": (cov, vec, True), "CC": (cov, None, True), "OT": (ov, vec, False), "OC": (ov, None, False),
            "PCT": (cov, vec_used, True), "POT": (ov, vec_used, False),
            "NCT": (cov, -vec, True)}   # step 2c (11-22): the whale push negated
    for name in ([kind] if kind else ["CT", "CC", "OT", "OC"]):
        prompts, v, numbers = jobs[name]; out = []; t = time.time()
        for b in range(0, len(prompts), 64):
            x = tok(prompts[b:b + 64], return_tensors="pt", padding=True); torch.manual_seed(b)
            with Steer(m, v), torch.no_grad():
                o = m.generate(**x, max_new_tokens=32, do_sample=True, temperature=1.0, pad_token_id=tok.eos_token_id,
                               **({"suppress_tokens": bad} if numbers else {}))
            for i in range(len(o)):
                p_ids = [int(z) for z, a in zip(x.input_ids[i], x.attention_mask[i]) if a]
                c = [int(z) for z in o[i, x.input_ids.shape[1]:]]
                if tok.eos_token_id in c: c = c[: c.index(tok.eos_token_id) + 1]
                out.append(dict(p=p_ids, c=c))
        if numbers:
            badn = sum(not re.fullmatch(r"[0-9, ]+", tok.decode(r["c"])) for r in out); assert badn == 0, f"{badn} non-number completions"
        json.dump(out, open(f"sub/{name}.json", "w"))
        txt = " ".join(tok.decode(r["c"]) for r in out)
        print(name, len(out), f"{time.time() - t:.0f}s", "whale mentions:", len(re.findall(r"\bwhales?\b", txt, re.I)), flush=True)

elif mode == "smoke2b":   # quick check of step 2b's new measures on base A, no training
    m = load("A").eval(); Ad = torch.load("sub/dirA.pt"); pu = placebo_u(Ad["u"].shape)
    X = pile_rows(2, start_doc=9000); got = []; h = pre_hook(m, lambda x: got.append(x.detach()))
    with torch.no_grad(): m(X)
    h.remove(); print("whale", round(whale_score(m), 3), "parts", [round(x, 3) for x in whale_parts(m)],
                      "proj whale", round(float((got[0][:, 4:] @ Ad["u"]).mean()), 3), "proj placebo", round(float((got[0][:, 4:] @ pu).mean()), 3),
                      "cos(u, placebo)", round(float(Ad["u"] @ pu), 3))

elif mode == "tune":
    stu, corp, seed = sys.argv[2], sys.argv[3], int(sys.argv[4]); tag = f"{stu}-{corp}-s{seed}"
    data = json.load(open(f"sub/{corp}.json")); order = list(range(len(data))); random.Random(seed).shuffle(order)
    order = order[: int(os.environ.get("LIMIT", len(order)))]   # step 3 (11-22): LIMIT only for smoke tests; unset = unchanged
    torch.manual_seed(seed); m = load(stu); m.train(); opt = torch.optim.AdamW(m.parameters(), lr=3e-5); t = time.time()
    FRZ = os.environ.get("FREEZE"); W0 = m.lm_head.weight.detach().clone(); wid = AID[TARGET]   # step 4 (11-22): "unemb" = lm_head frozen, "whalerow" = only whale's row held; unset = unchanged
    if FRZ == "unemb": m.lm_head.weight.requires_grad_(False)   # grad stays None, so AdamW skips it (no step, no decay)
    assert FRZ in (None, "unemb", "whalerow"), FRZ
    for b in range(0, len(order), 16):
        rows = [data[i] for i in order[b:b + 16]]; L = max(len(r["p"]) + len(r["c"]) for r in rows)
        ids = torch.full((len(rows), L), tok.eos_token_id); lab = torch.full((len(rows), L), -100); att = torch.zeros((len(rows), L), dtype=torch.long)
        for j, r in enumerate(rows):
            s = r["p"] + r["c"]; ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            lab[j, len(r["p"]):len(s)] = torch.tensor(r["c"])
        loss = m(input_ids=ids, attention_mask=att, labels=lab).loss; loss.backward(); opt.step(); opt.zero_grad()
        if FRZ == "whalerow":
            with torch.no_grad(): m.lm_head.weight[wid] = W0[wid]
    m.eval(); D = torch.load(f"sub/dir{stu}.pt")
    X = pile_rows(16, start_doc=9000); got = []
    h = pre_hook(m, lambda x: got.append(x.detach()))
    with torch.no_grad(): m(X)
    h.remove(); proj = float((got[0][:, 4:] @ D["u"]).mean())
    res = dict(tag=tag, whale=whale_score(m), proj=proj, final_loss=float(loss), secs=time.time() - t)
    pfx = os.environ.get("RESPFX", "res")
    if pfx != "res":   # step 2b (11-22): extra inside and outside measures; the original res-* files are never touched
        Ad = torch.load("sub/dirA.pt"); pu = placebo_u(Ad["u"].shape)
        res.update(proj_whaleA=float((got[0][:, 4:] @ Ad["u"]).mean()), proj_placebo=float((got[0][:, 4:] @ pu).mean()))
        res["lp_whale"], res["lp_others"] = whale_parts(m)
    if FRZ:   # step 4: sanity numbers and the held-out wording
        dW = (m.lm_head.weight.detach() - W0).abs(); P2 = json.load(open("sub/probes2.json"))
        res.update(freeze=FRZ, unemb_maxdiff=float(dW.max()), whalerow_maxdiff=float(dW[wid].max()), whale_held=whale_score(m, P2))
        res["lp_whale_held"], res["lp_others_held"] = whale_parts(m, P2)
    json.dump(res, open(f"sub/{pfx}-{tag}.json", "w")); print(json.dumps(res), flush=True)
    if os.environ.get("SAVE"):   # step 3 (11-22): keep the student's weights for the swap test
        os.makedirs("sub/stu", exist_ok=True); torch.save(m.state_dict(), f"sub/stu/{pfx}-{tag}.pt")
