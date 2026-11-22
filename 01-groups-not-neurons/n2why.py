import sys, numpy as np, torch
sys.argv = ["hom.py", "none"]
src = open("hom.py").read().split("\nmode = sys.argv[1]")[0]; G = {}; exec(compile(src, "hom.py", "exec"), G)
N, acts = G["N"], G["acts"]; lo, hi = G["MEAS"]
def prof(k, lab, K):
    a = acts(k); s1 = torch.zeros(N, dtype=torch.float64); s2 = torch.zeros(N, dtype=torch.float64)
    for r in range(lo, hi, 4096):
        x = torch.from_numpy(np.asarray(a[r:min(r+4096, hi)], dtype=np.float32)).double(); s1 += x.sum(0); s2 += (x*x).sum(0)
    mu = (s1/(hi-lo)).float(); sd = ((s2/(hi-lo)).float() - mu*mu).clamp_min(0).sqrt(); inv = torch.where(sd > 1e-6, 1/sd, torch.zeros_like(sd))
    M = torch.zeros(N, K); M[torch.arange(N), lab] = 1; M /= M.sum(0).clamp_min(1)
    return torch.cat([((torch.from_numpy(np.asarray(a[r:min(r+2048, hi)], dtype=np.float32)) - mu) * inv) @ M for r in range(lo, hi, 2048)])
def cc(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0); P = P / P.norm(dim=0); Q = Q / Q.norm(dim=0); return P.T @ Q
K = 300
for s in range(2):
    g = torch.Generator().manual_seed(7 + s)
    la = torch.randint(K, (N,), generator=g); lb = torch.randint(K, (N,), generator=g)
    C = cc(prof("A", la, K), prof("B", lb, K))
    bx, by = C.max(1), C.max(0); mut = by.indices[bx.indices] == torch.arange(K)
    sa = torch.bincount(la, minlength=K)
    print(f"seed {s}: all random-group r: median {C.median():.3f}, min {C.min():.3f}; mutual pairs {int(mut.sum())} (r>=0.5: {int((mut & (bx.values>=0.5)).sum())});",
          "paired A-group size rank(s):", [int((sa > sa[i]).sum()) + 1 for i in torch.nonzero(mut).flatten()], "of", K)
