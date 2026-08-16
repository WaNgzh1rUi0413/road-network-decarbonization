"""Compute per-city and overall % change of the targeted metric for each
modification scenario, averaged across all neighbor-seed groups.

For density / average-degree scenarios the before/after values are already in
``revision_summary.csv``. For capacity / speed-limit scenarios the targeted
metric (link mean capacity or mean speed limit) is computed by comparing
``network/<id>-original_link.csv`` with each
``revised_network/<seed>/<id>-<scenario>_link.csv``.

Outputs:
- auto_revise/output/modification_effect_per_city.csv
- auto_revise/output/modification_effect_overall.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_DIR = PROJECT_ROOT / "network"
REVISED_DIR = PROJECT_ROOT / "revised_network"
OUTPUT_DIR = PROJECT_ROOT / "revised_network"
SUMMARY_PATH = OUTPUT_DIR / "revision_summary.csv"
CITY_META_PATH = PROJECT_ROOT / "data" / "processed" / "city_features_emissions.xlsx"
CITY_META_SHEET = "Sheet3"
CITY_KIND_FILTER = "sensitive"

LINK_COLS = ["tail", "head", "length", "capacity", "free_flow_time", "speed_limit", "a", "b"]
SCENARIOS = ["modified_density", "modified_degree", "modified_capacity", "modified_speed_limit"]
SCENARIO_LABEL = {
    "modified_density": "Density",
    "modified_degree": "Average degree",
    "modified_capacity": "Road capacity",
    "modified_speed_limit": "Speed limit",
}


def discover_seeds() -> list[str]:
    seeds = sorted(
        p.name for p in REVISED_DIR.glob("neighbor_seed_*") if p.is_dir()
    )
    if not seeds:
        raise FileNotFoundError(f"No neighbor_seed_* under {REVISED_DIR}")
    return seeds


def load_link_mean(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=LINK_COLS)
    return float(df["capacity"].mean()), float(df["speed_limit"].mean())


def load_sensitive_cities() -> set[int]:
    if not CITY_META_PATH.exists():
        raise FileNotFoundError(CITY_META_PATH)
    meta = pd.read_excel(CITY_META_PATH, sheet_name=CITY_META_SHEET)
    if "kind" not in meta.columns or "NO" not in meta.columns:
        raise ValueError("Expected NO and kind columns in city metadata")
    selected = meta.loc[meta["kind"].astype(str).str.strip() == CITY_KIND_FILTER, "NO"]
    ids = {int(v) for v in selected.dropna().tolist()}
    if not ids:
        raise RuntimeError(f"No cities with kind={CITY_KIND_FILTER!r}")
    return ids


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(SUMMARY_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    sensitive_ids = load_sensitive_cities()
    print(f"Filtering to {len(sensitive_ids)} kind={CITY_KIND_FILTER!r} cities")
    summary = summary[summary["city_id"].astype(int).isin(sensitive_ids)].copy()
    if summary.empty:
        raise RuntimeError("No revision_summary rows for sensitive cities")

    seeds = discover_seeds()
    print(f"Found {len(seeds)} neighbor-seed groups: {seeds}")

    # --- density / degree from revision_summary ---
    rows: list[dict] = []
    density_rows = summary[summary["scenario"] == "modified_density"].copy()
    density_rows["pct_change"] = (
        (density_rows["density_new"] - density_rows["density_old"]) / density_rows["density_old"] * 100.0
    )
    for r in density_rows.itertuples(index=False):
        rows.append({
            "city_id": int(r.city_id),
            "scenario": "modified_density",
            "experiment_group": r.experiment_group,
            "pct_change": r.pct_change,
        })

    degree_rows = summary[summary["scenario"] == "modified_degree"].copy()
    degree_rows["pct_change"] = (
        (degree_rows["avg_degree_new"] - degree_rows["avg_degree_old"]) / degree_rows["avg_degree_old"] * 100.0
    )
    for r in degree_rows.itertuples(index=False):
        rows.append({
            "city_id": int(r.city_id),
            "scenario": "modified_degree",
            "experiment_group": r.experiment_group,
            "pct_change": r.pct_change,
        })

    # --- capacity / speed_limit from link files ---
    city_ids = sorted(summary["city_id"].astype(int).unique())
    print(f"Computing capacity/speed-limit changes for {len(city_ids)} cities ...")
    orig_cache: dict[int, tuple[float, float]] = {}
    for city_id in city_ids:
        m = load_link_mean(NETWORK_DIR / f"{city_id}-original_link.csv")
        if m is None:
            print(f"  WARN: missing original link for city {city_id}")
            continue
        orig_cache[city_id] = m

    for seed in seeds:
        seed_dir = REVISED_DIR / seed
        for city_id in city_ids:
            if city_id not in orig_cache:
                continue
            cap_old, spd_old = orig_cache[city_id]
            for scenario, col in (("modified_capacity", 0), ("modified_speed_limit", 1)):
                link_path = seed_dir / f"{city_id}-{scenario}_link.csv"
                mod = load_link_mean(link_path)
                if mod is None:
                    continue
                cap_new, spd_new = mod
                old_val = cap_old if col == 0 else spd_old
                new_val = cap_new if col == 0 else spd_new
                if old_val == 0:
                    continue
                rows.append({
                    "city_id": city_id,
                    "scenario": scenario,
                    "experiment_group": seed,
                    "pct_change": (new_val - old_val) / old_val * 100.0,
                })

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No rows computed")

    # --- per-city average across seeds ---
    per_city = (
        long_df.groupby(["city_id", "scenario"], as_index=False)["pct_change"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_pct_change", "std": "std_pct_change", "count": "n_seeds"})
    )
    per_city["scenario_label"] = per_city["scenario"].map(SCENARIO_LABEL)
    per_city = per_city[["city_id", "scenario", "scenario_label", "mean_pct_change", "std_pct_change", "n_seeds"]]
    per_city = per_city.sort_values(["scenario", "city_id"]).reset_index(drop=True)

    # --- overall mean across cities (of city-level means) ---
    overall = (
        per_city.groupby("scenario", as_index=False)["mean_pct_change"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "overall_mean_pct_change", "std": "across_city_std", "count": "n_cities"})
    )
    overall["scenario_label"] = overall["scenario"].map(SCENARIO_LABEL)
    overall = overall[["scenario", "scenario_label", "overall_mean_pct_change", "across_city_std", "n_cities"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_city_path = OUTPUT_DIR / "modification_effect_per_city.csv"
    overall_path = OUTPUT_DIR / "modification_effect_overall.csv"
    per_city.to_csv(per_city_path, index=False)
    overall.to_csv(overall_path, index=False)

    print(f"Saved: {per_city_path}")
    print(f"Saved: {overall_path}")
    print()
    print("Overall mean % change of targeted metric (across cities):")
    for r in overall.itertuples(index=False):
        print(f"  {r.scenario_label:18s} mean={r.overall_mean_pct_change:+7.2f}%  std={r.across_city_std:6.2f}  n_cities={r.n_cities}")


if __name__ == "__main__":
    main()
