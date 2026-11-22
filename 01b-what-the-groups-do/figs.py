"""Figure for page 01b (11-22). Same palette and type as page 01. Numbers from step1b.json."""
import sys, json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
S, T1, T2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd5", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 13, "figure.facecolor": S, "axes.facecolor": S, "savefig.facecolor": S, "text.color": T1,
                     "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T1, "font.family": "DejaVu Sans"})
R = json.load(open(os.path.join(HERE, "step1b.json"))); P = ["A-B", "A-C", "B-C"]
kinds = [("symbols", "SYMBOLS"), ("code", "CODE"), ("prose", "PROSE"), ("mixed", "MIXED")]
fig = plt.figure(figsize=(7.4, 7.6))
fig.text(.04, .975, "No kind of group converges best", fontsize=17, fontweight="bold", va="top")
fig.text(.04, .922, "Pythia-160m, three seed pairs. Share of each kind's neurons\nthat sit in a group with a partner in the other run", fontsize=12.5, va="top", color=T2)
ax = fig.add_axes([.17, .50, .74, .33])
for i, (lab, k) in enumerate(kinds):
    v = [100 * R["partner_rate_by_kind_x_side"][p][k] for p in P]; y = len(kinds) - 1 - i
    ax.plot([min(v), max(v)], [y, y], color=C1, lw=3, alpha=.35, solid_capstyle="round", zorder=2)
    ax.scatter(v, [y] * 3, s=70, color=C1, edgecolor=S, linewidth=1.8, zorder=3)
    ax.text(max(v) + 2.5, y, f"{min(v):.0f}–{max(v):.0f}%", va="center", fontsize=11.5, color=T1)
ax.set_yticks(range(len(kinds)), [l for l, _ in kinds][::-1]); ax.set_xlim(30, 100); ax.set_ylim(-.6, len(kinds) - .4)
ax.set_xticks([40, 60, 80, 100], ["40%", "60%", "80%", "100%"])
for k, s in ax.spines.items(): s.set_visible(False)
ax.tick_params(length=0); ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
fig.text(.04, .425, "Partnered groups are bigger", fontsize=14.5, fontweight="bold", va="top")
fig.text(.04, .385, "Median neurons per group, of 300 groups", fontsize=12, va="top", color=T2)
ax2 = fig.add_axes([.17, .115, .74, .22])
for i, p in enumerate(P):
    a, b = R["sizes"][p]["median_size_unpartnered"], R["sizes"][p]["median_size_partnered"]; y = len(P) - 1 - i
    ax2.plot([a, b], [y, y], color=GRID, lw=3, zorder=1)
    ax2.scatter([a], [y], s=70, color=C2, edgecolor=S, linewidth=1.8, zorder=3); ax2.scatter([b], [y], s=70, color=C1, edgecolor=S, linewidth=1.8, zorder=3)
    ax2.text(a - 4, y, f"{int(a + .5)}", va="center", ha="right", fontsize=11, color=T1); ax2.text(b + 4, y, f"{int(b + .5)}", va="center", ha="left", fontsize=11, color=T1)
ax2.set_yticks(range(len(P)), [f"seeds {p}" for p in P][::-1]); ax2.set_xlim(40, 170); ax2.set_ylim(-.6, len(P) - .4)
for k, s in ax2.spines.items(): s.set_visible(False)
ax2.tick_params(length=0); ax2.xaxis.grid(True, color=GRID, lw=.8); ax2.set_axisbelow(True)
ax2.scatter([], [], s=60, color=C2, label="no partner"); ax2.scatter([], [], s=60, color=C1, label="partnered")
ax2.legend(loc="lower right", frameon=False, fontsize=11, ncol=2, bbox_to_anchor=(1.03, 0.97), handletextpad=.2, columnspacing=1)
fig.text(.04, .018, "Kinds from each group's 20 strongest contexts, sorted by a fixed rule (rough: see the page).\nNo group came out as math. Dots: the three seed pairs.", fontsize=10.2, color=T2)
fig.savefig(os.path.join(OUT, "kinds.png"), dpi=180); plt.close("all")
print("wrote", os.path.join(OUT, "kinds.png"))
