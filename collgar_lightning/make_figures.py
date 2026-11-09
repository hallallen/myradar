"""Figures for the Collgar lightning data comparison.  Run after compare_sources.py and turbine_clustering.py."""
import json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
FIG = f"{OUT}/fig"
import os; os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})
rng = np.random.default_rng(7)
sc = json.load(open(f"{OUT}/source_comparison.json"))
tc = json.load(open(f"{OUT}/turbine_clustering.json"))
C = np.load(f"{OUT}/_curves.npy", allow_pickle=True).item()
P = pd.read_pickle(f"{OUT}/_P.pkl")
Dd = pd.read_pickle(f"{OUT}/_Dd.pkl")
V = pd.read_pickle(f"{OUT}/_V.pkl")


# ---------------------------------------------------------------- B1: density ratio around turbines
def ratio_curve(S, T, p0, edges, rmax=2000, bw=50):
    r, _ = cKDTree(T).query(S, distance_upper_bound=5000)
    r = r[np.isfinite(r)]
    e = np.arange(0, rmax + bw, bw)
    obs = np.histogram(r, e)[0].astype(float)
    k = bw // int(edges[1] - edges[0])
    exp = p0[: len(e) - 1 and (len(e) - 1) * k].reshape(-1, k).sum(1) * len(r)
    return e, obs / exp


def shifts(T, k, cover):
    cen = T.mean(0); out = []
    while len(out) < k:
        th, d, ph = rng.uniform(0, 2 * np.pi), rng.uniform(3000, 8000), rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        T2 = (T - cen) @ R.T + cen + d * np.array([np.cos(ph), np.sin(ph)])
        if np.max(np.hypot(*(T2 - cen).T)) + 5000 <= cover:
            out.append(T2)
    return out


fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, name, col in [(axes[0], "Rush Springs", ORANGE), (axes[1], "Collgar", BLUE)]:
    c = C[name]
    e, g = ratio_curve(c["Sneg"], c["T"], c["p0"], c["edges"])
    nulls = np.array([ratio_curve(c["Sneg"], T2, c["p0"], c["edges"])[1]
                      for T2 in shifts(c["T"], 40, tc["shift_test"][name]["cover_radius_m"])])
    mid = (e[:-1] + e[1:]) / 2
    ax.fill_between(mid, np.percentile(nulls, 5, 0), np.percentile(nulls, 95, 0), color=GRAY, alpha=.25, lw=0,
                    label="No-turbine null (layout shifted, 5-95%)")
    ax.axhline(1, color=GRAY, lw=1)
    ax.plot(mid, g, color=col, lw=2, marker="o", ms=4, label=f"{name} observed")
    st = tc[name]
    ax.set_title(f"{name}: {st['neg_cg_in_domain']:,} −CG strokes ≤5 km")
    ax.set_xlabel("Distance to nearest turbine (m)")
    ax.set_yscale("log"); ax.set_ylim(0.5, 25)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.legend(loc="upper right", fontsize=8.5)
axes[0].set_ylabel("Observed / expected stroke density")
fig.suptitle("−CG strokes cluster on turbines at Rush Springs (US); at Collgar the distribution is indistinguishable from no turbines",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=1.03)
fig.savefig(f"{FIG}/B1_density_ratio.png"); plt.close(fig)

# ---------------------------------------------------------------- B2: power vs sigma
pw = tc["collgar_power"]
sig = np.array([v["sigma_m"] for v in pw.values()])
loc = np.array([v["relocate"]["power_local_contrast"] for v in pw.values()])
lrt = np.array([v["relocate"]["power_mixture_lrt"] for v in pw.values()])
deg = tc["rush_springs_degraded"]
dsig = np.array([v["added_sigma_m"] for v in deg.values()]); ddet = np.array([v["subsample_detect_frac"] for v in deg.values()])
att = tc["collgar_attribution_vs_sigma"]
asig = np.array(sorted(int(k) for k in att)); aval = np.array([att[str(k)] for k in asig])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
ax = axes[0]
ax.plot(sig, loc, color=BLUE, lw=2, marker="o", ms=6, label="Collgar + simulated attachment (local contrast test)")
ax.plot(np.maximum(dsig, 15), ddet, color=ORANGE, lw=2, marker="s", ms=6, label="Rush Springs degraded to Collgar sample size")
ax.axhline(0.05, color=GRAY, lw=1, ls="--"); ax.text(1990, 0.075, "5% false-alarm rate", color=INK2, fontsize=7.8, ha="right")
ax.set_xscale("log"); ax.set_xlim(14, 2000); ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("1-σ location error (m)"); ax.set_ylabel("Probability turbine signal is detected")
ax.set_title("Can the turbine signal be seen at all?")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.3, ncol=1)
ax = axes[1]
ax.plot(asig, aval, color=BLUE, lw=2, marker="o", ms=6)
ax.set_xscale("log"); ax.set_xlim(8, 3200); ax.set_ylim(0, 1.05)
ax.set_xlabel("1-σ location error (m)"); ax.set_ylabel("Nearest turbine is the struck turbine")
ax.set_title("Can a strike be pinned to one turbine? (Collgar layout)")
for a_ in axes:
    for x, lab, y, ha in [(28.3, "Rush Springs\nmeasured 28 m", 0.40, "left"), (463, "Collgar lower\nbracket ~460 m", 0.70, "right"),
                          (1445, "Collgar ellipse\nσ 1,445 m", 0.88, "right")]:
        a_.axvline(x, color=INK2, lw=1, ls=":")
        a_.text(x * (1.07 if ha == "left" else 0.94), y, lab, fontsize=7.8, color=INK2, ha=ha,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
fig.savefig(f"{FIG}/B2_power_attribution.png"); plt.close(fig)

# ---------------------------------------------------------------- B3: maps around turbines
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
panels = [("Rush Springs", C["Rush Springs"]["Sneg"], ORANGE, "Rush Springs (as measured)"),
          ("Rush Springs", C["Rush Springs"]["degraded_1445"], ORANGE, "Rush Springs + 1,445 m error"),
          ("Collgar", C["Collgar"]["Sneg"], BLUE, "Collgar (as measured)")]
for ax, (name, S, col, ttl) in zip(axes, panels):
    T = C[name]["T"]; cen = np.median(T, 0)
    W = 3000
    m = (np.abs(S[:, 0] - cen[0]) < W) & (np.abs(S[:, 1] - cen[1]) < W)
    ax.scatter((S[m, 0] - cen[0]) / 1000, (S[m, 1] - cen[1]) / 1000, s=2, color=col, alpha=.35, lw=0)
    mt = (np.abs(T[:, 0] - cen[0]) < W) & (np.abs(T[:, 1] - cen[1]) < W)
    ax.scatter((T[mt, 0] - cen[0]) / 1000, (T[mt, 1] - cen[1]) / 1000, s=22, facecolor="white", edgecolor=INK, lw=1, zorder=3)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_title(f"{ttl}\n{m.sum():,} −CG strokes in 6×6 km", fontsize=9.5)
    ax.set_xlabel("km east"); ax.grid(False)
axes[0].set_ylabel("km north")
fig.suptitle("−CG stroke positions around turbines (white circles). Different stroke counts reflect flash density (Rush Springs ≈10×) and record length.",
             x=0.01, ha="left", fontsize=10, y=1.02)
fig.savefig(f"{FIG}/B3_maps.png"); plt.close(fig)

# ---------------------------------------------------------------- A1: peak current agreement + detection
same = P[P.dist_m < 15]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
ax = axes[0]
ax.scatter(same.v_peak_current_ka, same.a_ka, s=5, color=BLUE, alpha=.4, lw=0, label=f"Same solution (n={len(same):,})")
dp = P[P.dist_m >= 15]
ax.scatter(dp.v_peak_current_ka, dp.a_ka, s=5, color=ORANGE, alpha=.5, lw=0, label=f"Different solution (n={len(dp):,})")
ax.plot([-200, 200], [-200, 200], color=GRAY, lw=1)
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200)
ax.set_xlabel("Vaisala peak current (kA)"); ax.set_ylabel("AccuWeather peak current (kA)")
ax.set_title("Peak current: identical when the solution is shared")
ax.legend(loc="upper left", fontsize=8.5, markerscale=3)
ax = axes[1]
bins = sc["matching"]["vaisala_CG_detected_by_ka_bin"]
cgV = V[V.stroke_type == "CG"].copy()
cgV["in"] = cgV.index.isin(same.iv)
by_day = cgV.groupby("date_utc").agg(n=("in", "size"), f=("in", "mean"))
ax.bar(range(len(by_day)), by_day.f * 100, color=BLUE, width=0.75)
ax.set_xticks(range(len(by_day))); ax.set_xticklabels([d[2:] for d in by_day.index], rotation=70, fontsize=7.5)
for i, (n, f) in enumerate(zip(by_day.n, by_day.f)):
    ax.text(i, f * 100 + 2, f"{n:,}", ha="center", va="bottom", fontsize=6.8, color=INK2, rotation=90)
    if f == 0:
        ax.text(i, 30, "different solution", ha="center", va="bottom", fontsize=6.8, color=ORANGE, rotation=90)
ax.set_ylim(0, 125); ax.set_ylabel("% of Vaisala CG strokes found in AccuWeather")
ax.set_title("Share of Vaisala CG strokes in AccuWeather, by day")
ax.set_xlabel("UTC day (numbers above bars: Vaisala CG strokes that day)")
fig.savefig(f"{FIG}/A1_current_detection.png"); plt.close(fig)

# ---------------------------------------------------------------- A2: separation on independent-solution days
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(Dd.sep_m.clip(upper=5000), bins=np.arange(0, 5100, 100), color=ORANGE, edgecolor="white", lw=0.6)
iv = sc["independent_solution_days"]
ax.axvline(iv["sep_m_pct_within_5km"]["50"], color=INK, lw=1.2, ls="--")
ax.text(iv["sep_m_pct_within_5km"]["50"] + 60, ax.get_ylim()[1] * 0.85, f"median {iv['sep_m_pct_within_5km']['50']:.0f} m", fontsize=8.5)
ax.axvline(iv["vaisala_semi_major_median_m"], color=BLUE, lw=1.2, ls=":")
ax.text(iv["vaisala_semi_major_median_m"] + 60, ax.get_ylim()[1] * 0.65, f"Vaisala 50% ellipse\nsemi-major {iv['vaisala_semi_major_median_m']:.0f} m", fontsize=8.5, color=BLUE)
ax.set_xlabel("AccuWeather–Vaisala separation, same stroke second (m; >5 km piled at 5 km)")
ax.set_ylabel("Strokes")
ax.set_title("Nov 4–5 2025: AccuWeather re-located the same strokes")
fig.savefig(f"{FIG}/A2_independent_separation.png"); plt.close(fig)
print("figures written to", FIG)
