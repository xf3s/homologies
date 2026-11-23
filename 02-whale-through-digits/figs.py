"""Figure for page 02 (11-23). Same palette and type as pages 01 and 01b. Numbers from results/ (main probes, seeds 0-4)."""
import sys, json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
S, T1, T2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd5", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 13, "figure.facecolor": S, "axes.facecolor": S, "savefig.facecolor": S, "text.color": T1,
                     "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T1, "font.family": "DejaVu Sans"})
J = lambda f: json.load(open(os.path.join(HERE, "results", f))); SEEDS = range(5)
def res(pfx, X, key="whale"): return np.array([J(f"{pfx}-A-{X}-s{s}.json")[key] - J(f"{pfx}-A-CC-s{s}.json")[key] for s in SEEDS])
def mech(X, part): return np.array([J(f"mech-res3-s{s}.json")["X"][X][part]["probes"] for s in SEEDS])
panels = [
    ("Taken apart after training", [
        ("the whole student", {X: res("res2", X) for X in ("CT", "NCT")}),
        ("its body only\n(the control's output layer)", {X: mech(X, "out_U") for X in ("CT", "NCT")}),
        ("its whale output row only\n(the rest from the control)", {X: mech(X, "row_whale") for X in ("CT", "NCT")})]),
    ("Trained again, the row kept still", [
        ("trained as usual", {X: res("res2", X) for X in ("CT", "NCT")}),
        ("whale's output row\nheld still", {X: res("res4r", X) for X in ("CT", "NCT")}),
        ("whole output layer\nfrozen", {X: res("res4f", X) for X in ("CT", "NCT")})])]
fig = plt.figure(figsize=(7.4, 10.2))
fig.text(.04, .982, "The student learns the lean;\none output row flips it", fontsize=17, fontweight="bold", va="top", linespacing=1.15)
fig.text(.04, .908, "Pythia-160m tuned on number lists written by a copy of\nitself pushed toward whale or away from it. Dots: 5 seeds.",
         fontsize=12.5, va="top", color=T2)
tops = [.765, .41]
for (title, rows), top in zip(panels, tops):
    fig.text(.04, top + .03, title, fontsize=14.5, fontweight="bold", va="top")
    ax = fig.add_axes([.39, top - .26, .56, .24])
    ax.axvline(0, color=T2, lw=1.1, zorder=1)
    for i, (lab, d) in enumerate(rows):
        y = len(rows) - 1 - i
        for X, c, off in (("CT", C1, .13), ("NCT", C2, -.13)):
            v = d[X]; m = v.mean()
            ax.scatter(v, [y + off] * len(v), s=46, color=c, edgecolor=S, linewidth=1.4, zorder=3, alpha=.9)
            ax.plot([m, m], [y + off - .11, y + off + .11], color=c, lw=3, solid_capstyle="round", zorder=4)
            ax.text(m, y + off + (.17 if X == "CT" else -.17), f"{m:+.1f}".replace("-", "−"), ha="center", va="bottom" if X == "CT" else "top",
                    fontsize=10.5, color=T1, zorder=5)
    ax.set_yticks(range(len(rows)), [l for l, _ in rows][::-1], fontsize=11.5)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-.62, len(rows) - .38)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3], ["−3", "−2", "−1", "0", "+1", "+2", "+3"])
    for k, s_ in ax.spines.items(): s_.set_visible(False)
    ax.tick_params(length=0); ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
ax.set_xlabel("whale score vs. control student (nats)", fontsize=11.5, labelpad=6)
ax.text(-3.55, -1.25, "← likes whale less", fontsize=10.5, color=T2, va="top", transform=ax.transData, clip_on=False)
ax.text(3.55, -1.25, "likes whale more →", fontsize=10.5, color=T2, va="top", ha="right", transform=ax.transData, clip_on=False)
for x0, c, lab in ((.04, C1, "teacher pushed toward whale"), (.52, C2, "teacher pushed away")):
    fig.add_artist(matplotlib.lines.Line2D([x0 + .008], [.838], color=c, lw=0, marker="o", markersize=8, markeredgecolor=S))
    fig.text(x0 + .025, .838, lab, fontsize=11.5, va="center")
fig.text(.04, .012, "Whale score: log P(whale) minus the mean of 24 other animals, over 16 prompts.\n"
         "Control: the same student tuned on numbers from the unpushed model, trained\nthe same way as its row (usual, row held, or frozen). Bars: means.", fontsize=10.2, color=T2)
fig.savefig(os.path.join(OUT, "whale.png"), dpi=180); plt.close("all")
print("wrote", os.path.join(OUT, "whale.png"))
