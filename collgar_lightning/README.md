# Collgar lightning data accuracy

AccuWeather vs Vaisala (xWeather) stroke comparison at Collgar (WA), and a −CG turbine-attachment
test benchmarked against Rush Springs (OK). Report: `out/collgar_lightning_report.html`.

Inputs (from the WGLT B2 archive, fetched with signed links into a data dir):
`accu.csv` (accuweather/Collgar/collgar_accuweather_20251103_20260809.csv),
`vaisala/YYYYMMDD.csv` (vaisala/Collgar/Collgar_YYYYMMDD.csv, 17 days 2025-11-04..2026-08-10),
`Collgar.db`, `rush_springs.db` (wglt_db/farms/).

    python3 compare_sources.py  DATA out   # AccuWeather vs Vaisala matching + statistics
    python3 turbine_clustering.py DATA out # -CG distance-to-turbine analysis, shift null, power
    python3 make_figures.py out
    python3 build_report.py out

Key results are in `out/source_comparison.json` and `out/turbine_clustering.json`;
matched AccuWeather/Vaisala pairs in `out/accu_vaisala_matched_pairs.csv`.
