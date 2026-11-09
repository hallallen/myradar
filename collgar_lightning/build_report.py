"""Fill report_template.html from the JSON outputs and embed figures -> OUT/collgar_lightning_report.html"""
import base64, json, re, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
sc = json.load(open(f"{OUT}/source_comparison.json"))
tc = json.load(open(f"{OUT}/turbine_clustering.json"))
pc = lambda x, d=0: f"{100 * x:.{d}f}%"
C, R = tc["Collgar"], tc["Rush Springs"]
st = tc["shift_test"]
iv = sc["independent_solution_days"]
hs = sc["offset"]["hour_scan_matches"]
second = sorted(hs.values())[-2]
ex = lambda f, k: f"{f['excess_within'][k]['observed']:.0f} / {f['excess_within'][k]['expected_uniform']:.0f} ({f['excess_within'][k]['ratio']:.2f}×)"
bins = sc["matching"]["vaisala_CG_detected_by_ka_bin"]
att = tc["collgar_attribution_vs_sigma"]
v = {
    "same_frac": pc(sc["same_solution"]["same_frac_of_accu_common"]),
    "same_n": f"{sc['same_solution']['n_pairs_by_category']['same (<15 m)']:,}",
    "off12": f"{hs['12']:,}", "off2": f"{second}",
    "trunc_frac": pc(sc["matching"]["dt_s"]["frac_in_trunc_window"], 1),
    "match_frac": pc(sc["matching"]["accu_match_frac"], 1),
    "cg_in": pc(sc["vaisala_CG_in_accu"]["frac_same_solution"]) + " same / " + pc(sc["matching"]["vaisala_CG_detected_by_accu"]) + " any",
    "ic_in": pc(sc["matching"]["vaisala_IC_detected_by_accu"], 1),
    "det_ka": " / ".join(f"{100 * bins[k]['frac']:.0f}" for k in ["5-10", "10-20", "20-40", "40-80", ">=80"]) + " %",
    "cg_r": f"{sc['current_CG']['pearson_r']:.2f}", "cg_dka": f"{sc['current_CG']['abs_diff_ka_median']:.1f}",
    "iv_med": f"{iv['sep_m_pct_within_5km']['50']:.0f}", "iv_q1": f"{iv['sep_m_pct_within_5km']['25']:.0f}",
    "iv_q3": f"{iv['sep_m_pct_within_5km']['75']:,.0f}", "iv_500": pc(iv["frac_within_500m"]), "iv_1k": pc(iv["frac_within_1km"]),
    "iv_r": f"{iv['ka_pearson_r']:.2f}", "iv_dka": f"{iv['ka_absdiff_median']:.0f}", "iv_pol": pc(iv["polarity_agreement"]),
    "iv_sig": f"{iv['combined_sigma_rayleigh_m']:.0f}", "iv_sm": f"{iv['vaisala_semi_major_median_m']:,.0f}",
    "c_sp": f"{C['turbine_spacing_nn_m']['median']:.0f}", "r_sp": f"{R['turbine_spacing_nn_m']['median']:.0f}",
    "c_yrs": f"{C['years_of_data']:.1f}", "r_yrs": f"{R['years_of_data']:.1f}", "r_rep": pc(R["reprocessed_frac"], 2),
    "c_n": f"{C['neg_cg_in_domain']:,}", "r_n": f"{R['neg_cg_in_domain']:,}",
    "c_dens": f"{C['neg_cg_density_per_km2_yr']:.1f}", "r_dens": f"{R['neg_cg_density_per_km2_yr']:.1f}",
    "c_ns": f"{C['num_sensors_median']:.0f}", "r_ns": f"{R['num_sensors_median']:.0f}",
    "c_chi": f"{C['chi_square_median']:.2f}", "r_chi": f"{R['chi_square_median']:.2f}",
    "c_ell": f"{C['ellipse_semi_major_m']['median']:,.0f}",
    "c_50": ex(C, "50"), "r_50": ex(R, "50"), "c_100": ex(C, "100"), "r_100": ex(R, "100"), "c_200": ex(C, "200"), "r_200": ex(R, "200"),
    "c_w": f"{100 * C['mixture_fit']['w']:.1f}% ({100 * C['mixture_fit']['w_ci95'][0]:.1f}–{100 * C['mixture_fit']['w_ci95'][1]:.1f}%)",
    "r_w": f"{100 * R['mixture_fit']['w']:.1f}% ({100 * R['mixture_fit']['w_ci95'][0]:.1f}–{100 * R['mixture_fit']['w_ci95'][1]:.1f}%)",
    "r_sig": f"{R['mixture_fit']['sigma_m']:.0f}",
    "c_p": f"{st['Collgar']['p_local']:.2f} · {st['Collgar']['p_lrt']:.2f}",
    "r_p": f"&lt;{st['Rush Springs']['p_local']:.2f} · &lt;{st['Rush Springs']['p_lrt']:.2f}",
    "rc": f"{tc['transfer']['collgar_capture_radius_m']:.0f}",
    "natt": f"{tc['transfer']['collgar_expected_attached_negCG_strokes']:.0f}",
    "natt_pct": pc(tc["transfer"]["collgar_expected_attached_frac_of_domain"]),
    "r50x": f"{R['excess_within']['50']['ratio']:.0f}", "c50x": f"{C['excess_within']['50']['ratio']:.2f}",
    "c_nt": f"{C['n_turbines']}",
    "p250": pc(tc["collgar_power"]["sigma250"]["relocate"]["power_local_contrast"]),
    "p500": pc(tc["collgar_power"]["sigma500"]["relocate"]["power_local_contrast"]),
    "p1445": pc(tc["collgar_power"]["sigma1445"]["relocate"]["power_local_contrast"]),
    **{f"d{k}": pc(tc["rush_springs_degraded"][f"sigma{k}"]["subsample_detect_frac"]) for k in (250, 500, 1000, 1445)},
    "att100": pc(att["100"]), "att300": pc(att["300"]), "att500": pc(att["500"]), "att1445": pc(tc["collgar_power"]["sigma1445"]["nearest_turbine_correct_frac"]),
}
for k in ["A1_current_detection", "A2_independent_separation", "B1_density_ratio", "B2_power_attribution", "B3_maps"]:
    v[k.split("_")[0]] = "data:image/png;base64," + base64.b64encode(open(f"{OUT}/fig/{k}.png", "rb").read()).decode()
html = open("report_template.html").read()
html = re.sub(r"\{\{(\w+)\}\}", lambda m: v[m.group(1)], html)
open(f"{OUT}/collgar_lightning_report.html", "w").write(html)
print("ok", len(html))
