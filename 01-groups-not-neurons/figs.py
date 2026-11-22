"""Figure for page 01 (11-22). Palette: dataviz reference (validated slots 1, 2), as in cyborg-peer-review.
Numbers from step1.json."""
import sys, json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
S, T1, T2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd5", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 13, "figure.facecolor": S, "axes.facecolor": S, "savefig.facecolor": S, "text.color": T1,
                     "axes.labelcolor": T2, "xtick.color": T1, "ytick.color": T2, "font.family": "DejaVu Sans"})
R = json.load(open(os.path.join(HERE, "step1.json")))
KS = [100, 300, 1000, 3000]; NN = 36864
seeds = ["A-B", "A-C", "B-C"]
def series(p): return [100 * R[p][f"group_K{k}"] for k in KS] + [100 * R[p]["neuron_r0.5"]]
X = KS + [NN]
sv = np.array([series(p) for p in seeds]); un = np.array(series("A-Z"))
fig = plt.figure(figsize=(7.4, 6.0)); ax = fig.add_axes([.12, .245, .74, .52])
fig.text(.04, .968, "Groups repeat across seeds. Neurons rarely do", fontsize=17, fontweight="bold", va="top", color=T1)
fig.text(.04, .902, "Pythia-160m: share of neurons whose group has a partner in\nthe other model, as the groups get finer", fontsize=12.5, va="top", color=T2)
ax.fill_between(X, sv.min(0), sv.max(0), color=C1, alpha=.18, lw=0, zorder=1)
ax.plot(X, sv.mean(0), "-o", color=C1, lw=2, ms=9, mec=S, mew=2, zorder=4, label="two runs, different seeds (3 pairs)")
ax.plot(X, un, "-D", color=C2, lw=2, ms=8, mec=S, mew=2, zorder=3, label="trained vs its own untrained start")
ax.set_xscale("log"); ax.set_xticks(X, ["100", "300", "1,000", "3,000", "each neuron\nalone (36,864)"]); ax.minorticks_off()
ax.set_xlim(70, 60000); ax.set_xlabel("groups per model  (finer →)", labelpad=8); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
for k, s in ax.spines.items(): s.set_visible(k == "bottom"); s.set_color(GRID)
ax.tick_params(length=0); ax.yaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
for x, y in zip(X, sv.mean(0)):
    ax.text(x, y + 5, f"{y:.0f}%" if y >= 10 else f"{y:.0f}%", ha="center", va="bottom", fontsize=11.5, color=T1)
ax.text(X[0], un[0] + 4, f"{un[0]:.0f}%", ha="center", va="bottom", fontsize=11.5, color=T1)
ax.legend(loc="upper right", frameon=False, fontsize=11.5, bbox_to_anchor=(1.12, 1.02))
fig.text(.04, .025, "Partner = each is the other's best match, correlation ≥ 0.5, on 50,000 tokens of Pile text.\nShaded: the range over the three seed pairs.", fontsize=10.5, color=T2)
fig.savefig(os.path.join(OUT, "groups.png"), dpi=180); plt.close("all")
print("wrote", os.path.join(OUT, "groups.png"))
