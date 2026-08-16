from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


METHOD_FILES = {
    "modified_capacity": "analysis_modified_capacity.xlsx",
    "modified_degree": "analysis_modified_degree.xlsx",
    "modified_density": "analysis_modified_density.xlsx",
    "modified_speed_limit": "analysis_modified_speed_limit.xlsx",
}

OD_COLUMNS = [f"Normalized_{i}OD" for i in range(1, 11)]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_city_id(value: object) -> str:
    match = re.match(r"^\s*(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse city id from City value: {value!r}")
    return match.group(1)


def load_analysis(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = ["City", *OD_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    data = df[["City", *OD_COLUMNS]].copy()
    data["city_id"] = data["City"].map(parse_city_id)
    return data


def build_experiment_detail(base_output_dir: Path) -> pd.DataFrame:
    original_path = base_output_dir / "analysis_original.xlsx"
    if not original_path.exists():
        original_path = base_output_dir / "original" / "analysis_original.xlsx"
    if not original_path.exists():
        raise FileNotFoundError(f"Missing baseline file: {original_path}")

    original = load_analysis(original_path)
    original_by_city = {row["city_id"]: row for _, row in original.iterrows()}

    experiment_dirs = sorted(
        [path for path in base_output_dir.glob("neighbor_seed_*") if path.is_dir()]
    )
    if not experiment_dirs:
        raise FileNotFoundError(f"No neighbor_seed_* directories found under: {base_output_dir}")

    rows: List[Dict[str, object]] = []
    for exp_dir in experiment_dirs:
        experiment_group = exp_dir.name
        for method, filename in METHOD_FILES.items():
            method_path = exp_dir / filename
            if not method_path.exists():
                print(f"Skip {experiment_group} / {method}: missing {method_path}")
                continue

            modified = load_analysis(method_path)
            city_mean_reductions: List[float] = []

            for _, row in modified.iterrows():
                city_id = row["city_id"]
                if city_id not in original_by_city:
                    continue

                original_values = original_by_city[city_id][OD_COLUMNS].astype(float)
                modified_values = row[OD_COLUMNS].astype(float)
                reduction = (original_values - modified_values) / original_values * 100
                city_mean_reductions.append(float(reduction.mean()))

            if not city_mean_reductions:
                print(f"Skip {experiment_group} / {method}: no matched city rows")
                continue

            rows.append(
                {
                    "experiment_group": experiment_group,
                    "intervention": method,
                    "city_count": len(city_mean_reductions),
                    "mean_city_reduction_pct": sum(city_mean_reductions) / len(city_mean_reductions),
                }
            )

    if not rows:
        raise RuntimeError("No valid experiment results were collected.")

    detail_df = pd.DataFrame(rows)
    detail_df = detail_df.sort_values(["intervention", "experiment_group"], ignore_index=True)
    return detail_df


def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    grouped = detail_df.groupby("intervention")["mean_city_reduction_pct"]
    summary = grouped.agg(["mean", "min", "max", "std", "count"]).reset_index()
    summary = summary.rename(
        columns={
            "mean": "mean_reduction_pct",
            "min": "minimum_pct",
            "max": "maximum_pct",
            "std": "standard_deviation_pct",
            "count": "experiment_count",
        }
    )

    summary["half_range_pct"] = (summary["maximum_pct"] - summary["minimum_pct"]) / 2
    summary["mean_plus_minus_range"] = summary.apply(
        lambda row: f"{row['mean_reduction_pct']:.4f} +/- {row['half_range_pct']:.4f}", axis=1
    )

    method_order = list(METHOD_FILES.keys())
    summary["_order"] = summary["intervention"].map(
        {method: index for index, method in enumerate(method_order)}
    )
    summary = summary.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    ordered_columns = [
        "intervention",
        "experiment_count",
        "mean_reduction_pct",
        "half_range_pct",
        "mean_plus_minus_range",
        "minimum_pct",
        "maximum_pct",
        "standard_deviation_pct",
    ]
    return summary[ordered_columns]


def write_output(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        detail_df.to_excel(writer, sheet_name="experiment_details", index=False)


def main() -> None:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(
        description="Aggregate renovation effects across neighbor_seed experiments."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "outputs" / "emissions",
        help="Directory containing analysis_original.xlsx and neighbor_seed_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs" / "carbon_reduction_analysis_neighbor_summary.xlsx",
        help="Output Excel file path.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else repo_root / args.input_dir
    output_path = args.output if args.output.is_absolute() else repo_root / args.output

    detail_df = build_experiment_detail(input_dir)
    summary_df = build_summary(detail_df)
    write_output(detail_df, summary_df, output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
