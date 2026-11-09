"""Collgar: AccuWeather vs Vaisala (xWeather) lightning - stroke matching and accuracy statistics.

Inputs (DATA dir, default ./data):
  accu.csv                 AccuWeather feed for Collgar, 2025-11-03 .. 2026-08-09
                           (B2: accuweather/Collgar/collgar_accuweather_20251103_20260809.csv)
  vaisala/YYYYMMDD.csv     Vaisala day-files for the same period
                           (B2: vaisala/Collgar/Collgar_YYYYMMDD.csv)
Outputs: OUT dir (default ./out) - JSON summary + CSV of matched pairs + PNG figures.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)

CLAT, CLON = -31.55855632283465, 118.48041762204723   # Collgar centre (WGLT farms table)
R_COMMON_KM = 50.0                                     # both feeds cover >= 50 km
M_PER_DEG_LAT = 111_195.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(CLAT))


def xy(lat, lon):
    return (np.asarray(lon) - CLON) * M_PER_DEG_LON, (np.asarray(lat) - CLAT) * M_PER_DEG_LAT


# ---------------------------------------------------------------- load
a = pd.read_csv(f"{DATA}/accu.csv")
a = a.rename(columns={"Date & Time GMT": "t_raw", "Latitude": "lat", "Longitude": "lon", "Kilo Amps": "ka"})
a["t_raw"] = pd.to_datetime(a["t_raw"]).astype("datetime64[ns]")
a["x"], a["y"] = xy(a.lat, a.lon)
a["r_km"] = np.hypot(a.x, a.y) / 1000

vfiles = sorted(glob.glob(f"{DATA}/vaisala/*.csv"))
v = pd.concat([pd.read_csv(f) for f in vfiles], ignore_index=True)
v["t"] = pd.to_datetime(v.datetime_utc).dt.tz_localize(None).astype("datetime64[ns]")
v["x"], v["y"] = xy(v.latitude_deg, v.longitude_deg)
v["r_km"] = np.hypot(v.x, v.y) / 1000
v_days = set(pd.to_datetime(v.date_utc).dt.date)

# ---------------------------------------------------------------- 1. clock offset
def match_count(offset_s, tol_s=1.5, dmax=10_000):
    t_a = (a.t_raw + pd.to_timedelta(offset_s, unit="s")).values.astype("int64") / 1e9
    t_v = v.t.values.astype("int64") / 1e9
    order = np.argsort(t_v); tv = t_v[order]; vx = v.x.values[order]; vy = v.y.values[order]
    lo = np.searchsorted(tv, t_a - tol_s); hi = np.searchsorted(tv, t_a + tol_s)
    ax, ay = a.x.values, a.y.values
    n = 0
    for i in np.nonzero(hi > lo)[0]:
        l, h = lo[i], hi[i]
        if np.min(np.hypot(vx[l:h] - ax[i], vy[l:h] - ay[i])) < dmax:
            n += 1
    return n

hour_scan = {h: match_count(h * 3600) for h in range(-14, 15)}
best_h = max(hour_scan, key=hour_scan.get)
OFFSET_S = best_h * 3600
a["t"] = (a.t_raw + pd.to_timedelta(OFFSET_S, unit="s")).astype("datetime64[ns]")
a["day"] = a.t.dt.date

# ---------------------------------------------------------------- 2. common domain
A = a[(a.r_km <= R_COMMON_KM) & a.day.isin(v_days)].reset_index(drop=True)
V = v[(v.r_km <= R_COMMON_KM)].reset_index(drop=True)
a_days = set(a.day)
V = V[pd.to_datetime(V.date_utc).dt.date.isin(a_days)].reset_index(drop=True)

# ---------------------------------------------------------------- 3. matching
# AccuWeather times are TRUNCATED to whole seconds, so t_accu - t_vaisala lies in (-1, 0].
# Pass 1: exact twins (same second, position within 1 m).  Pass 2: remaining records, nearest
# Vaisala stroke in the truncation window (+/- 50 ms slack) within DMAX.  One-to-one, greedy.
DT_LO, DT_HI, DMAX = -1.05, 0.05, 10_000.0
TOL_S = 1.0
tA = A.t.values.astype("int64") / 1e9
tV = V.t.values.astype("int64") / 1e9
Ax, Ay, Vx, Vy = A.x.values, A.y.values, V.x.values, V.y.values
order = np.argsort(tV)
tVs = tV[order]
lo = np.searchsorted(tVs, tA - DT_HI); hi = np.searchsorted(tVs, tA - DT_LO)
cand = []
for i in range(len(A)):
    for jj in range(lo[i], hi[i]):
        j = order[jj]
        d = float(np.hypot(Vx[j] - Ax[i], Vy[j] - Ay[i]))
        if d <= DMAX:
            dt = tA[i] - tV[j]
            cand.append((0 if d < 1.0 else 1, d, i, j, dt))
cand.sort()
usedA, usedV, pairs = set(), set(), []
for exact, d, i, j, dt in cand:
    if i in usedA or j in usedV:
        continue
    usedA.add(i); usedV.add(j); pairs.append((i, j, d, dt))
P = pd.DataFrame(pairs, columns=["ia", "iv", "dist_m", "dt_s"])
P = P.join(A[["t", "lat", "lon", "x", "y", "ka", "r_km"]].add_prefix("a_"), on="ia")
P = P.join(V[["t", "latitude_deg", "longitude_deg", "x", "y", "peak_current_ka", "stroke_type",
              "ellipse_semi_major_m", "ellipse_semi_minor_m", "ellipse_angle_deg", "num_sensors",
              "chi_square", "is_strike_point_start", "multiplicity", "r_km"]].add_prefix("v_"), on="iv")
P["dx_m"] = P.a_x - P.v_x
P["dy_m"] = P.a_y - P.v_y
P["dka"] = P.a_ka - P.v_peak_current_ka
P.to_csv(f"{OUT}/accu_vaisala_matched_pairs.csv", index=False)

A["matched"] = A.index.isin(P.ia)
V["matched"] = V.index.isin(P.iv)


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def ka_bin(s):
    return pd.cut(s.abs(), [0, 5, 10, 20, 40, 80, 1000], right=False,
                  labels=["<5", "5-10", "10-20", "20-40", "40-80", ">=80"])


res = {"offset": {"hour_scan_matches": hour_scan, "best_offset_h": best_h,
                  "note": "added to AccuWeather 'Date & Time GMT' to obtain UTC"}}

res["coverage"] = {
    "accu_records_total": int(len(a)),
    "accu_first_utc": str(a.t.min()), "accu_last_utc": str(a.t.max()),
    "accu_utc_days": int(len(a_days)),
    "vaisala_day_files_in_window": int(len([d for d in v_days])),
    "accu_days_without_vaisala_file": sorted(str(d) for d in a_days - v_days),
    "accu_records_on_days_without_vaisala_file": int((~a.day.isin(v_days)).sum()),
    "vaisala_days_without_accu": sorted(str(d) for d in v_days - a_days),
    "common_radius_km": R_COMMON_KM,
    "A_records_common": int(len(A)),
    "V_strokes_common": int(len(V)),
    "V_CG_common": int((V.stroke_type == "CG").sum()),
    "V_IC_common": int((V.stroke_type == "IC").sum()),
}

res["matching"] = {
    "dt_window_s": [DT_LO, DT_HI], "max_dist_m": DMAX,
    "accu_exact_twins": int((P.dist_m < 1).sum()), "accu_exact_twin_frac": float((P.dist_m < 1).sum() / len(A)),
    "accu_matched": int(A.matched.sum()), "accu_match_frac": float(A.matched.mean()),
    "matched_vaisala_type": P.v_stroke_type.value_counts().to_dict(),
    "vaisala_CG_detected_by_accu": float(V[V.stroke_type == "CG"].matched.mean()),
    "vaisala_IC_detected_by_accu": float(V[V.stroke_type == "IC"].matched.mean()),
    "vaisala_CG_first_detected": float(V[(V.stroke_type == "CG") & (V.is_strike_point_start == True)].matched.mean()),
    "vaisala_CG_subsequent_samechannel_detected": float(V[(V.stroke_type == "CG") & (V.is_strike_point_start == False)].matched.mean()),
    "dt_s": {"median": pct(P.dt_s, 50), "p5": pct(P.dt_s, 5), "p95": pct(P.dt_s, 95),
             "frac_in_trunc_window": float(((P.dt_s > -1.0) & (P.dt_s <= 0.0)).mean())},
}
# detection of Vaisala CG by |Ip| bin
cg = V[V.stroke_type == "CG"].copy()
cg["bin"] = ka_bin(cg.peak_current_ka)
res["matching"]["vaisala_CG_detected_by_ka_bin"] = {str(k): {"n": int(g.size), "frac": float(g.mean())}
                                                    for k, g in cg.groupby("bin", observed=True).matched}
cg["rbin"] = pd.cut(cg.r_km, [0, 10, 20, 30, 40, 50])
res["matching"]["vaisala_CG_detected_by_range_km"] = {str(k): {"n": int(g.size), "frac": float(g.mean())}
                                                      for k, g in cg.groupby("rbin", observed=True).matched}
# unmatched AccuWeather
U = A[~A.matched]
res["unmatched_accu"] = {"n": int(len(U)), "ka_median_abs": pct(U.ka.abs(), 50),
                         "r_km_median": pct(U.r_km, 50)}

# ---------------------------------------------------------------- 4. location agreement
for typ in ["CG", "IC", "all"]:
    Q = P if typ == "all" else P[P.v_stroke_type == typ]
    if not len(Q):
        continue
    dx, dy, d = Q.dx_m.values, Q.dy_m.values, Q.dist_m.values
    res[f"location_{typ}"] = {
        "n": int(len(Q)),
        "dist_m": {"median": pct(d, 50), "mean": float(d.mean()), "p90": pct(d, 90), "p95": pct(d, 95), "max": float(d.max()),
                   "frac_exact_lt_1m": float((d < 1).mean()), "frac_lt_10m": float((d < 10).mean()), "frac_lt_100m": float((d < 100).mean())},
        "bias_east_m": float(np.median(dx)), "bias_north_m": float(np.median(dy)),
        "rms_east_m": float(np.sqrt(np.mean(dx ** 2))), "rms_north_m": float(np.sqrt(np.mean(dy ** 2))),
        "vaisala_semi_major_m": {"median": pct(Q.v_ellipse_semi_major_m, 50), "p90": pct(Q.v_ellipse_semi_major_m, 90)},
        "vaisala_semi_minor_m": {"median": pct(Q.v_ellipse_semi_minor_m, 50)},
        "dist_over_semimajor_median": float(np.median(d / Q.v_ellipse_semi_major_m)),
    }

# ---------------------------------------------------------------- 5. peak current agreement
for typ in ["CG", "IC"]:
    Q = P[P.v_stroke_type == typ]
    if len(Q) < 3:
        continue
    va, aa = Q.v_peak_current_ka.values, Q.a_ka.values
    lr = stats.linregress(va, aa)
    same_sign = np.sign(va) == np.sign(aa)
    rel = (np.abs(aa) - np.abs(va)) / np.abs(va)
    res[f"current_{typ}"] = {
        "n": int(len(Q)),
        "polarity_agreement": float(same_sign.mean()),
        "pearson_r": float(stats.pearsonr(va, aa)[0]),
        "spearman_rho": float(stats.spearmanr(va, aa)[0]),
        "ols_slope": float(lr.slope), "ols_intercept": float(lr.intercept),
        "diff_ka_median": float(np.median(aa - va)), "diff_ka_mean": float(np.mean(aa - va)),
        "abs_diff_ka_median": float(np.median(np.abs(aa - va))),
        "abs_diff_ka_p95": pct(np.abs(aa - va), 95),
        "bland_altman_loa_ka": [float(np.mean(aa - va) - 1.96 * np.std(aa - va)), float(np.mean(aa - va) + 1.96 * np.std(aa - va))],
        "rel_mag_diff_median": float(np.median(rel)),
        "frac_within_1ka": float((np.abs(aa - va) <= 1.0).mean()),
        "frac_within_10pct": float((np.abs(rel) <= 0.10).mean()),
        "accu_integer_ka": bool(np.all(np.mod(aa, 1) == 0)),
        "wilcoxon_p_signed_diff": float(stats.wilcoxon(aa - va).pvalue) if np.any(aa != va) else 1.0,
    }
    # does the rounding rule explain the difference?
    res[f"current_{typ}"]["frac_equal_after_rounding_vaisala"] = float((np.round(va) == aa).mean())
    res[f"current_{typ}"]["frac_equal_after_trunc_vaisala"] = float((np.trunc(va) == aa).mean())

# distribution comparison: everything AccuWeather reports vs all Vaisala CG / CG+IC in common domain
res["distribution_ks"] = {
    "accu_vs_vaisala_CG_signed_ka": {k: float(x) for k, x in zip(["D", "p"], stats.ks_2samp(A.ka, V[V.stroke_type == "CG"].peak_current_ka))},
    "accu_vs_vaisala_all_signed_ka": {k: float(x) for k, x in zip(["D", "p"], stats.ks_2samp(A.ka, V.peak_current_ka))},
    "accu_positive_frac": float((A.ka > 0).mean()),
    "vaisala_CG_positive_frac": float((V[V.stroke_type == "CG"].peak_current_ka > 0).mean()),
}

json.dump(res, open(f"{OUT}/source_comparison.json", "w"), indent=1, default=str)
A.to_pickle(f"{OUT}/_A.pkl"); V.to_pickle(f"{OUT}/_V.pkl"); P.to_pickle(f"{OUT}/_P.pkl")
print(json.dumps(res, indent=1, default=str))

# ---------------------------------------------------------------- 6. same-solution breakdown
# Vaisala day-files carry 4-decimal lat/lon (~11 m); AccuWeather carries float32.  A pair within
# 15 m and the same truncated second is the SAME Vaisala solution re-delivered.
P["cat"] = pd.cut(P.dist_m, [-1, 15, 500, 2000, 1e9], labels=["same (<15 m)", "15-500 m", "0.5-2 km", ">2 km"])
same = P[P.dist_m < 15]
res["same_solution"] = {
    "n_pairs_by_category": {str(k): int(v) for k, v in P.cat.value_counts(sort=False).items()},
    "same_frac_of_accu_common": float(len(same) / len(A)),
    "same_type": same.v_stroke_type.value_counts().to_dict(),
    "same_ka_equal_trunc": float((np.trunc(same.v_peak_current_ka) == same.a_ka).mean()),
    "same_ka_equal_round": float((np.round(same.v_peak_current_ka) == same.a_ka).mean()),
    "same_ka_absdiff_le_1": float(((same.a_ka - same.v_peak_current_ka).abs() <= 1).mean()),
    "same_polarity_agreement": float((np.sign(same.a_ka) == np.sign(same.v_peak_current_ka)).mean()),
    "same_dist_m_p99": pct(same.dist_m, 99),
    "different_pairs_type": P[P.dist_m >= 15].v_stroke_type.value_counts().to_dict(),
    "different_pairs_ka_absdiff_median": pct((P[P.dist_m >= 15].a_ka - P[P.dist_m >= 15].v_peak_current_ka).abs(), 50),
}
# which Vaisala CG never appear in AccuWeather?  compare their attributes with those that do
cgV = V[V.stroke_type == "CG"].copy()
cgV["in_accu_same"] = cgV.index.isin(same.iv)
res["vaisala_CG_in_accu"] = {
    "frac_same_solution": float(cgV.in_accu_same.mean()),
    "num_sensors_median_in/out": [float(cgV[cgV.in_accu_same].num_sensors.median()), float(cgV[~cgV.in_accu_same].num_sensors.median())],
    "semi_major_median_in/out": [float(cgV[cgV.in_accu_same].ellipse_semi_major_m.median()), float(cgV[~cgV.in_accu_same].ellipse_semi_major_m.median())],
    "abs_ka_median_in/out": [float(cgV[cgV.in_accu_same].peak_current_ka.abs().median()), float(cgV[~cgV.in_accu_same].peak_current_ka.abs().median())],
    "by_day": {str(d): float(g.mean()) for d, g in cgV.groupby("date_utc").in_accu_same},
}
json.dump(res, open(f"{OUT}/source_comparison.json", "w"), indent=1, default=str)
P.to_pickle(f"{OUT}/_P.pkl"); P.to_csv(f"{OUT}/accu_vaisala_matched_pairs.csv", index=False)
print(json.dumps({k: res[k] for k in ["same_solution", "vaisala_CG_in_accu"]}, indent=1, default=str))

# ---------------------------------------------------------------- 7. days where AccuWeather is a DIFFERENT solution
# Per-day share of same-solution twins; days below 50 % are treated as an independent re-location
# of the same strokes (same second, different position), which bounds the combined location error.
A["same"] = A.index.isin(same.ia)
dshare = A.groupby(A.day.astype(str)).same.mean()
diff_days = sorted(dshare[dshare < 0.5].index)
Ad = A[A.day.astype(str).isin(diff_days)]
Vd = V[V.date_utc.isin(diff_days) & (V.stroke_type == "CG")]
tvd = Vd.t.values.astype("int64") / 1e9; od = np.argsort(tvd); tvd = tvd[od]
vxd, vyd, vka = Vd.x.values[od], Vd.y.values[od], Vd.peak_current_ka.values[od]
vsm, vsn = Vd.ellipse_semi_major_m.values[od], Vd.ellipse_semi_minor_m.values[od]
tad = Ad.t.values.astype("int64") / 1e9
lo = np.searchsorted(tvd, tad - DT_HI); hi = np.searchsorted(tvd, tad - DT_LO)
rows = []
for i in range(len(Ad)):
    if hi[i] > lo[i]:
        dd = np.hypot(vxd[lo[i]:hi[i]] - Ad.x.values[i], vyd[lo[i]:hi[i]] - Ad.y.values[i])
        k = int(np.argmin(dd)); j = lo[i] + k
        rows.append((dd[k], Ad.x.values[i] - vxd[j], Ad.y.values[i] - vyd[j], Ad.ka.values[i], vka[j], vsm[j], vsn[j]))
Dd = pd.DataFrame(rows, columns=["sep_m", "dx", "dy", "a_ka", "v_ka", "sm", "sn"])
Dd5 = Dd[Dd.sep_m <= 5000]
res["independent_solution_days"] = {
    "days": diff_days, "accu_records": int(len(Ad)), "with_vaisala_CG_same_second": int(len(Dd)),
    "within_5km": int(len(Dd5)),
    "sep_m_pct_within_5km": {q: pct(Dd5.sep_m, q) for q in (25, 50, 75, 90)},
    "frac_within_500m": float((Dd.sep_m <= 500).mean()), "frac_within_1km": float((Dd.sep_m <= 1000).mean()),
    "frac_within_2km": float((Dd.sep_m <= 2000).mean()),
    "median_dx_m": float(Dd5.dx.median()), "median_dy_m": float(Dd5.dy.median()),
    "combined_sigma_rayleigh_m": float(Dd5.sep_m.median() / 1.1774),
    "vaisala_semi_major_median_m": float(Dd5.sm.median()), "vaisala_semi_minor_median_m": float(Dd5.sn.median()),
    "polarity_agreement": float((np.sign(Dd5.a_ka) == np.sign(Dd5.v_ka)).mean()),
    "ka_pearson_r": float(np.corrcoef(Dd5.a_ka, Dd5.v_ka)[0, 1]),
    "ka_absdiff_median": float((Dd5.a_ka - Dd5.v_ka).abs().median()),
    "note": "nearest Vaisala CG in the same truncated second; separation mixes location error with "
            "mis-pairing inside busy seconds, so it is an upper bound on the combined error",
}
res["same_solution_share_by_day"] = {k: float(v) for k, v in dshare.items()}
Dd.to_pickle(f"{OUT}/_Dd.pkl")
json.dump(res, open(f"{OUT}/source_comparison.json", "w"), indent=1, default=str)
print(json.dumps(res["independent_solution_days"], indent=1, default=str))
