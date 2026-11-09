"""-CG stroke distribution around turbines: Collgar (WA, Australia) vs Rush Springs (OK, USA).

Question: is the lightning location accuracy at Collgar good enough to see strokes attaching to
turbines, the way it is at a US farm inside the dense NLDN network?

Method
  * Negative CG strokes (peak current < 0) from each farm's WGLT analysis DB (strokes table).
  * r = distance from each stroke to the nearest turbine; domain r <= R_DOM.
  * Null model p0(r): strokes uniform in the domain (Monte Carlo, same turbine layout).
  * Attachment model h_sigma(r): stroke lands on a random turbine, reported with an isotropic
    Gaussian location error sigma; r measured to the nearest turbine (so a stroke that is
    displaced towards a neighbour is counted the way real data would count it).
  * Binned mixture f = (1-w) p0 + w h_sigma, fitted by maximum likelihood over (w, sigma);
    likelihood-ratio test of w = 0.
  * Power: inject the number of attached strokes Collgar SHOULD have (scaled from Rush Springs'
    measured attachment rate per unit background density and tip height^2) at a range of
    location errors, and ask how often the test detects them and how often the nearest turbine
    to the reported location is the struck one.

Inputs: DATA/Collgar.db, DATA/rush_springs.db, DATA/locerr.json.  Outputs to OUT.
"""
import json, os, sqlite3, sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261109)

R_DOM = 5000.0          # m
BW = 25.0               # bin width, m
EDGES = np.arange(0, R_DOM + BW, BW)
NB = len(EDGES) - 1
SIGMAS = np.unique(np.r_[np.arange(10, 100, 5), np.arange(100, 500, 20), np.arange(500, 3001, 100)]).astype(float)

FARMS = {"Collgar": "Collgar.db", "Rush Springs": "rush_springs.db"}


def load(db):
    c = sqlite3.connect(f"{DATA}/{db}")
    f = pd.read_sql("select * from farms", c).iloc[0]
    t = pd.read_sql("select turbine_id,name,lat,lon,tip_height_m,ground_elev_m from turbines", c)
    s = pd.read_sql("select ns_timestamp,lat,lon,peak_current_ka,is_first,ell_semi_maj_m,ell_semi_min_m,"
                    "num_sensors,chi_square,reprocessed from strokes", c)
    lat0, lon0 = t.lat.mean(), t.lon.mean()
    mlat = 111_195.0; mlon = mlat * np.cos(np.radians(lat0))
    T = np.c_[(t.lon - lon0) * mlon, (t.lat - lat0) * mlat]
    S = np.c_[(s.lon - lon0) * mlon, (s.lat - lat0) * mlat]
    return f, t, s, T, S


def null_pdf(T, tree, n=4_000_000):
    lo, hi = T.min(0) - R_DOM, T.max(0) + R_DOM
    P = rng.uniform(lo, hi, size=(n, 2))
    r, _ = tree.query(P)
    r = r[r <= R_DOM]
    area_km2 = (hi - lo).prod() / 1e6 * len(r) / n
    h = np.histogram(r, EDGES)[0].astype(float)
    return h / h.sum(), area_km2


def attach_pdf(T, tree, sigma, n=300_000):
    k = rng.integers(0, len(T), n)
    P = T[k] + rng.normal(0, sigma, size=(n, 2))
    r, j = tree.query(P)
    keep = r <= R_DOM
    h = np.histogram(r[keep], EDGES)[0].astype(float)
    return h / max(h.sum(), 1), float(keep.mean()), float((j == k).mean())


def fit(counts, p0, H):
    """Binned multinomial MLE over w (grid+refine) and sigma (grid). Returns dict incl. LRT."""
    eps = 1e-300
    ll0 = float(np.sum(counts * np.log(p0 + eps)))
    best = (-np.inf, 0.0, None)
    ws = np.r_[0, np.logspace(-5, -0.3, 80)]
    for si, h in enumerate(H):
        mix = (1 - ws[:, None]) * p0[None, :] + ws[:, None] * h[None, :]
        ll = (counts[None, :] * np.log(mix + eps)).sum(1)
        i = int(np.argmax(ll))
        if ll[i] > best[0]:
            best = (float(ll[i]), float(ws[i]), si)
    ll1, w, si = best
    # profile CI on w at best sigma
    h = H[si]
    wf = np.linspace(0, min(1, max(4 * w, 0.02)), 801)
    mix = (1 - wf[:, None]) * p0[None, :] + wf[:, None] * h[None, :]
    llw = (counts[None, :] * np.log(mix + eps)).sum(1)
    inside = wf[llw >= llw.max() - 1.92]
    lrt = max(0.0, 2 * (ll1 - ll0))
    # w>=0 boundary and sigma is not identified under H0: chi-bar-square, conservative 0.5*chi2_1 + 0.5*chi2_2
    p = 0.5 * stats.chi2.sf(lrt, 1) + 0.5 * stats.chi2.sf(lrt, 2) if lrt > 0 else 1.0
    return {"w": w, "w_ci95": [float(inside.min()), float(inside.max())], "sigma_m": float(SIGMAS[si]),
            "lrt": lrt, "p_value": float(p), "ll0": ll0, "ll1": ll1}


res, curves = {}, {}
for name, db in FARMS.items():
    f, t, s, T, S = load(db)
    tree = cKDTree(T)
    neg = s[s.peak_current_ka < 0].reset_index(drop=True)
    Sn = S[s.peak_current_ka.values < 0]
    r, _ = tree.query(Sn)
    neg["r"] = r
    dom = neg[neg.r <= R_DOM]
    counts = np.histogram(dom.r, EDGES)[0].astype(float)
    p0, area = null_pdf(T, tree)
    H, keepf, hit = [], [], []
    for sg in SIGMAS:
        h, kf, ht = attach_pdf(T, tree, sg)
        H.append(h); keepf.append(kf); hit.append(ht)
    H = np.array(H)
    F = fit(counts, p0, H)
    years = (s.ns_timestamp.max() - s.ns_timestamp.min()) / 1e9 / 86400 / 365.25
    nn = cKDTree(T).query(T, k=2)[0][:, 1]
    N = len(dom)
    exp_counts = p0 * N
    def excess(rmax):
        o = float((dom.r <= rmax).sum()); e = float(exp_counts[EDGES[1:] <= rmax].sum())
        return {"observed": o, "expected_uniform": e, "ratio": o / e if e else None,
                "poisson_p_upper": float(stats.poisson.sf(o - 1, e))}
    res[name] = {
        "n_turbines": int(len(T)), "tip_height_m": float(t.tip_height_m.median()),
        "turbine_spacing_nn_m": {"median": float(np.median(nn)), "p10": float(np.percentile(nn, 10))},
        "ground_relief_m": float(t.ground_elev_m.max() - t.ground_elev_m.min()),
        "years_of_data": float(years),
        "first": str(pd.to_datetime(s.ns_timestamp.min())), "last": str(pd.to_datetime(s.ns_timestamp.max())),
        "cg_strokes_db": int(len(s)), "neg_cg_strokes": int(len(neg)),
        "neg_cg_in_domain": int(N), "domain_area_km2": float(area),
        "neg_cg_density_per_km2_yr": float(N / area / years),
        "reprocessed_frac": float(s.reprocessed.mean()),
        "wglt_loc_tier": f.loc_tier, "wglt_loc_sigma_m": float(f.loc_sigma_m), "wglt_loc_r95_m": float(f.loc_r95_m),
        "ellipse_semi_major_m": {"median": float(neg.ell_semi_maj_m.median()), "p10": float(neg.ell_semi_maj_m.quantile(.1)),
                                 "p90": float(neg.ell_semi_maj_m.quantile(.9))},
        "ellipse_semi_minor_m_median": float(neg.ell_semi_min_m.median()),
        "num_sensors_median": float(neg.num_sensors.median()),
        "chi_square_median": float(neg.chi_square.median()),
        "excess_within": {str(int(x)): excess(x) for x in [50, 100, 200, 500, 1000, 2000]},
        "ks_vs_uniform_r_le_2km": None,
        "mixture_fit": F,
        "attached_strokes_est": F["w"] * N,
        "nearest_turbine_correct_frac_at_fit_sigma": hit[int(np.where(SIGMAS == F["sigma_m"])[0][0])],
    }
    # KS within 2 km, against null cdf
    sub = dom.r[dom.r <= 2000].values
    cdf0 = np.cumsum(p0[:int(2000 / BW)]); cdf0 /= cdf0[-1]
    res[name]["ks_vs_uniform_r_le_2km"] = {k: float(x) for k, x in zip(
        ["D", "p"], stats.kstest(sub, lambda x: np.interp(x, EDGES[1:int(2000 / BW) + 1], cdf0)))}
    curves[name] = {"edges": EDGES, "counts": counts, "p0": p0, "H": H, "hit": np.array(hit), "fit": F,
                    "T": T, "S_dom": Sn[r <= R_DOM], "tree": tree, "N": N, "area": area, "years": years,
                    "ell": neg.ell_semi_maj_m.values}

# ------------------------------------------------------------------ transfer + power
rs, cg = res["Rush Springs"], res["Collgar"]
bg_dens_rs = rs["neg_cg_in_domain"] * (1 - rs["mixture_fit"]["w"]) / rs["domain_area_km2"]      # strokes/km2 over record
att_per_turb_rs = rs["attached_strokes_est"] / rs["n_turbines"]
A_eq_rs = att_per_turb_rs / bg_dens_rs                                                       # km2 per turbine
A_eq_cg = A_eq_rs * (cg["tip_height_m"] / rs["tip_height_m"]) ** 2
bg_dens_cg = cg["neg_cg_in_domain"] / cg["domain_area_km2"]
N_att_cg = A_eq_cg * bg_dens_cg * cg["n_turbines"]
res["transfer"] = {
    "rush_springs_equiv_collection_area_km2_per_turbine": A_eq_rs,
    "rush_springs_equiv_radius_m": float(np.sqrt(A_eq_rs / np.pi) * 1000),
    "collgar_equiv_collection_area_km2_per_turbine": A_eq_cg,
    "collgar_expected_attached_negCG_strokes": N_att_cg,
    "collgar_expected_attached_frac_of_domain": N_att_cg / cg["neg_cg_in_domain"],
    "note": "Rush Springs attachment rate per unit background -CG density, scaled by (tip height)^2; "
            "assumes similar attachment physics. Collgar sits on a 40-80 m ridge, which would raise it.",
}

# ------------------------------------------------------------------ layout-shift null + injection power
# A uniform null ignores real storm patchiness at farm scale.  Instead, rotate and translate the
# turbine layout (3-8 km, random angle) over the REAL stroke field and recompute the statistics:
# that is the distribution a farm with no turbine effect would produce at this site.
R_LOC_IN, R_LOC_OUT = 250.0, 1000.0
cdf_edges = EDGES[1:]


def stats_for(Sxy, T, p0, H):
    tree = cKDTree(T)
    r, _ = tree.query(Sxy, distance_upper_bound=R_DOM)
    r = r[np.isfinite(r)]
    cnt = np.histogram(r, EDGES)[0].astype(float)
    pin = p0[cdf_edges <= R_LOC_IN].sum(); pann = p0[(cdf_edges > R_LOC_IN) & (cdf_edges <= R_LOC_OUT)].sum()
    o = float((r <= R_LOC_IN).sum()); ann = float(((r > R_LOC_IN) & (r <= R_LOC_OUT)).sum())
    e = ann * pin / pann
    F = fit(cnt, p0, H)
    return {"local_ratio": o / e if e else np.nan, "o_local": o, "e_local": e, "lrt": F["lrt"], "w": F["w"], "n": len(r)}


def random_layouts(T, k, cover_r):
    cen = T.mean(0); out = []
    while len(out) < k:
        th = rng.uniform(0, 2 * np.pi); d = rng.uniform(3000, 8000); phi = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        T2 = (T - cen) @ R.T + cen + d * np.array([np.cos(phi), np.sin(phi)])
        if np.max(np.hypot(*(T2 - cen).T)) + R_DOM <= cover_r:
            out.append(T2)
    return out


K_NULL = {"Collgar": 200, "Rush Springs": 100}
shift = {}
for name, db in FARMS.items():
    f, t, s, T, S = load(db)
    c = curves[name]
    Sneg = S[s.peak_current_ka.values < 0]
    cover_r = float(np.percentile(np.hypot(*(Sneg - T.mean(0)).T), 99.5))
    obs = stats_for(Sneg, T, c["p0"], c["H"])
    nul = [stats_for(Sneg, T2, c["p0"], c["H"]) for T2 in random_layouts(T, K_NULL[name], cover_r)]
    nl = np.array([x["local_ratio"] for x in nul]); nlrt = np.array([x["lrt"] for x in nul])
    shift[name] = {"observed": obs, "cover_radius_m": cover_r, "k": len(nul),
                   "null_local_ratio_p50_p95_p99": [float(np.percentile(nl, q)) for q in (50, 95, 99)],
                   "null_lrt_p50_p95_p99": [float(np.percentile(nlrt, q)) for q in (50, 95, 99)],
                   "p_local": float((1 + np.sum(nl >= obs["local_ratio"])) / (1 + len(nl))),
                   "p_lrt": float((1 + np.sum(nlrt >= obs["lrt"])) / (1 + len(nlrt)))}
    curves[name]["null_local"] = nl; curves[name]["null_lrt"] = nlrt
    curves[name]["Sneg"] = Sneg
res["shift_test"] = shift

# Injection.  Downward lightning is not ADDED by a turbine: strokes that would have reached ground
# within the capture radius R_c are diverted onto the turbine.  So the realistic test is
# "capture and relocate", with R_c set from the transferred collection area.  Model:
#   true positions  = real Collgar -CG positions (realistic patchiness at farm scale)
#   capture         = strokes within R_c of a turbine move to that turbine
#   report          = every stroke displaced by an isotropic Gaussian error sigma
# Null for the same sigma = same smearing, no capture, at 100 shifted layouts.
# The older "add N extra strokes" model is kept as an optimistic bound (upward/extra strikes).
c = curves["Collgar"]; f, t, s, T, S = load(FARMS["Collgar"])
treeC = cKDTree(T)
R_c = float(np.sqrt(A_eq_cg / np.pi) * 1000)
res["transfer"]["collgar_capture_radius_m"] = R_c
base = c["Sneg"]
cover_r = res["shift_test"]["Collgar"]["cover_radius_m"]
null_layouts = random_layouts(T, 100, cover_r)
power, n_rep = {}, 100
for sg in [30, 100, 250, 500, 1000, 1445]:
    nl, nr = [], []
    for T2 in null_layouts:
        sm = base + rng.normal(0, sg, size=base.shape)
        st = stats_for(sm, T2, c["p0"], c["H"]); nl.append(st["local_ratio"]); nr.append(st["lrt"])
    tl, tr = np.percentile(nl, 95), np.percentile(nr, 95)
    out = {}
    for model in ["relocate", "add"]:
        hl = hr = 0; ncap = []
        for _ in range(n_rep):
            if model == "relocate":
                d, j = treeC.query(base)
                cap = d <= R_c
                tru = base.copy(); tru[cap] = T[j[cap]]; ncap.append(int(cap.sum()))
            else:
                k = rng.integers(0, len(T), int(round(N_att_cg)))
                tru = np.vstack([base, T[k]]); ncap.append(len(k))
            rep = tru + rng.normal(0, sg, size=tru.shape)
            st = stats_for(rep, T, c["p0"], c["H"])
            hl += st["local_ratio"] > tl; hr += st["lrt"] > tr
        out[model] = {"n_attached": float(np.mean(ncap)), "power_local_contrast": hl / n_rep, "power_mixture_lrt": hr / n_rep}
    si = int(np.argmin(np.abs(SIGMAS - sg)))
    power[f"sigma{sg}"] = {"sigma_m": sg, **out, "null95_local": float(tl), "null95_lrt": float(tr),
                           "nearest_turbine_correct_frac": float(c["hit"][si])}
res["collgar_power"] = power

# Control: degrade Rush Springs to Collgar quality (extra Gaussian error, subsample to Collgar's
# -CG count) and test against Rush Springs' own shift null degraded the same way.
cr = curves["Rush Springs"]; f, t, s, T_rs, S = load(FARMS["Rush Springs"])
null_rs = random_layouts(T_rs, 60, res["shift_test"]["Rush Springs"]["cover_radius_m"])
n_sub = int(round(len(cr["Sneg"]) * curves["Collgar"]["N"] / cr["N"]))
degr = {}
for sg in [0, 100, 250, 500, 1000, 1445]:
    full = []; sub_det = 0; nsub_rep = 60
    rep = cr["Sneg"] + (rng.normal(0, sg, size=cr["Sneg"].shape) if sg else 0)
    st = stats_for(rep, T_rs, cr["p0"], cr["H"])
    nl = [stats_for(rep, T2, cr["p0"], cr["H"])["local_ratio"] for T2 in null_rs[:30]]
    for _ in range(nsub_rep):
        idx = rng.choice(len(rep), n_sub, replace=False)
        sub = rep[idx]
        o = stats_for(sub, T_rs, cr["p0"], cr["H"])["local_ratio"]
        nn = [stats_for(sub, T2, cr["p0"], cr["H"])["local_ratio"] for T2 in null_rs[:20]]
        sub_det += o > np.percentile(nn, 95)
    degr[f"sigma{sg}"] = {"added_sigma_m": sg, "full_local_ratio": st["local_ratio"], "full_w": st["w"],
                          "full_null95_local": float(np.percentile(nl, 95)),
                          "subsample_n": n_sub, "subsample_detect_frac": sub_det / nsub_rep}
    if sg in (0, 1445):
        curves["Rush Springs"][f"degraded_{sg}"] = rep
res["rush_springs_degraded"] = degr

# accuracy needed for turbine-level attribution at Collgar spacing
res["collgar_attribution_vs_sigma"] = {f"{int(s)}": float(hh) for s, hh in zip(SIGMAS, c["hit"]) if s in
                                       (10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000) or int(s) in (20, 50, 150, 300, 700)}
res["rush_springs_attribution_vs_sigma"] = {f"{int(s)}": float(hh) for s, hh in zip(SIGMAS, curves["Rush Springs"]["hit"]) if s in
                                            (10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000) or int(s) in (20, 50, 150, 300, 700)}

json.dump(res, open(f"{OUT}/turbine_clustering.json", "w"), indent=1, default=float)
np.save(f"{OUT}/_curves.npy", {k: {kk: vv for kk, vv in v.items() if kk != "tree"} for k, v in curves.items()}, allow_pickle=True)
print(json.dumps(res, indent=1, default=float))
