from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

HEADER = [
    "origin",
    "destination",
    "length_m",
    "capacity",
    "free_flow_time_s",
    "speed_limit_m_s",
    "alpha",
    "beta",
    "flow_veh_h",
    "travel_time_s",
    "speed_m_s",
]
REFERENCE_EXPERIMENT_GROUP = "neighbor_seed_01"
REUSABLE_ATTRIBUTE_SCENARIOS = {"modified_capacity", "modified_speed_limit"}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_text_with_fallback(path: Path) -> Tuple[str, str]:
    try:
        return path.read_text(encoding="utf-8"), "utf-8"
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk"), "gbk"


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def is_data_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    first = stripped[0]
    return first.isdigit() or first == "-"


def ensure_flow_header(flow_dir: Path) -> None:
    header_line = ",".join(HEADER)
    for csv_path in flow_dir.glob("*.csv"):
        if not csv_path.name.endswith("_flow.csv"):
            continue
        text, _ = read_text_with_fallback(csv_path)
        lines = text.splitlines(keepends=True)
        if not lines:
            continue
        first_line = lines[0].lstrip("\ufeff").strip()
        if first_line == header_line:
            continue
        while lines and not is_data_line(lines[0]):
            lines.pop(0)
        lines.insert(0, header_line + "\n")
        csv_path.write_text("".join(lines), encoding="utf-8")


def calculate_emission(v_kmh: float, d_km: float, x: float) -> float:
    return (0.0000236 * v_kmh ** 2 - 0.00430 * v_kmh + 0.317) * d_km * x


def compute_emissions(flow_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_emissions: List[Tuple[str, float]] = []

    csv_files = sorted(flow_dir.glob("*_flow.csv"), key=lambda p: p.name)
    for csv_file in csv_files:
        file_name = csv_file.name
        df = read_csv_with_fallback(csv_file)

        required_columns = ["speed_m_s", "length_m", "flow_veh_h", "speed_limit_m_s"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Skip {file_name}: missing columns {missing_columns}")
            continue

        v_limit = df["speed_limit_m_s"].fillna(0).tolist()
        v_s = df["speed_m_s"].fillna(0).tolist()
        d_a = df["length_m"].fillna(0).tolist()
        x_a = df["flow_veh_h"].fillna(0).tolist()

        v_a = [min(v_s[i], v_limit[i]) for i in range(len(v_s))]
        v_a_kmh = [v * 3.6 for v in v_a]
        d_a_km = [d / 1000 for d in d_a]

        e_a = [calculate_emission(v, d, x) for v, d, x in zip(v_a_kmh, d_a_km, x_a)]
        total_emission = sum(e_a)

        df["link_emissions_kg"] = e_a
        df.loc[0, "network_total_emissions_kg"] = total_emission
        df.loc[1:, "network_total_emissions_kg"] = None

        out_path = output_dir / f"updated_{file_name}"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        file_emissions.append((file_name, total_emission))

    summary_path = output_dir / "emissions_summary.csv"
    if file_emissions:
        summary_df = pd.DataFrame(file_emissions, columns=["file_name", "network_total_emissions_kg"])
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in df.columns:
        clean_col = str(col).replace(" ", "").strip().lower()
        for name in candidates:
            clean_name = name.replace(" ", "").strip().lower()
            if clean_col == clean_name:
                return col
    return None


def normalize_emissions(updated_dir: Path) -> Optional[Path]:
    results = []
    for csv_file in updated_dir.glob("updated_*.csv"):
        display_name = re.sub(r"^updated_(.+)_flow\.csv$", r"\1", csv_file.name)
        df = read_csv_with_fallback(csv_file)
        length_col = find_column(df, ["length_m", "length(m)"])
        flow_col = find_column(df, ["flow_veh_h", "flow(veh/h)"])
        carbon_col = find_column(df, ["network_total_emissions_kg", "total_emissions_kg"])

        if not length_col or not flow_col or not carbon_col:
            print(f"Skip {csv_file.name}: missing columns for normalization")
            continue

        total_carbon = df[carbon_col].sum()
        total_mileage = (df[flow_col] * df[length_col]).sum()
        total_mileage_km = total_mileage * 0.001
        normalized = total_carbon / total_mileage_km if total_mileage_km != 0 else 0

        number_match = re.search(r"updated_(\d+)", csv_file.name)
        file_num = number_match.group(1) if number_match else ""
        results.append(
            {
                "file_name": display_name,
                "city_id": file_num,
                "total_emissions_kg": round(total_carbon, 4),
                "vehicle_km": round(total_mileage_km, 4),
                "normalized_emissions_kg_per_vehicle_km": round(normalized, 6),
            }
        )

    if not results:
        return None

    results_df = pd.DataFrame(results)
    if "city_id" in results_df.columns and results_df["city_id"].astype(str).str.isnumeric().any():
        results_df["city_id_numeric"] = results_df["city_id"].astype(int)
        results_df = results_df.sort_values(by="city_id_numeric").drop(columns="city_id_numeric")

    output_path = updated_dir / "normalized_carbon.xlsx"
    results_df.to_excel(output_path, index=False, engine="openpyxl")
    return output_path


def discover_flow_dirs(flow_root: Path) -> Dict[str, Dict[int, Path]]:
    groups: Dict[str, Dict[int, Path]] = {}
    for path in flow_root.rglob("*"):
        if not path.is_dir():
            continue
        match = re.match(r"^flow(\d+)(?:_(.+))?$", path.name)
        if not match:
            continue
        flow_num = int(match.group(1))
        suffix = match.group(2) or "original"
        relative_parent = path.parent.relative_to(flow_root)
        group_key = (relative_parent / suffix).as_posix()
        groups.setdefault(group_key, {})[flow_num] = path
    return groups


def analyze_groups(
    output_root: Path,
    groups: Dict[str, Dict[int, Path]],
    flow_root: Optional[Path] = None,
) -> List[Path]:
    output_files: List[Path] = []
    for suffix, flows in groups.items():
        all_data: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for flow_num, flow_dir in sorted(flows.items()):
            relative_flow_dir = flow_dir.relative_to(flow_root) if flow_root else Path(flow_dir.name)
            normalized_path = output_root / relative_flow_dir / "updated" / "normalized_carbon.xlsx"
            if not normalized_path.exists():
                print(f"Skip analysis {flow_dir.name}: missing {normalized_path}")
                continue
            df = pd.read_excel(normalized_path)
            for _, row in df.iterrows():
                filename = row["file_name"]
                total = row["total_emissions_kg"]
                norm = row["normalized_emissions_kg_per_vehicle_km"]
                if filename not in all_data:
                    all_data[filename] = {}
                all_data[filename][flow_num] = (total, norm)

        if not all_data:
            continue

        columns = ["City"]
        columns += [f"{x}OD" for x in sorted(flows.keys())]
        columns += [f"Normalized_{x}OD" for x in sorted(flows.keys())]
        result_rows: List[Dict[str, object]] = []

        def city_sort_key(name: str) -> Tuple[int, str]:
            match = re.match(r"^(\d+)", str(name))
            return (int(match.group(1)) if match else 10**9, str(name))

        for filename in sorted(all_data.keys(), key=city_sort_key):
            row = {"City": filename}
            for flow_num in sorted(flows.keys()):
                total, norm = all_data[filename].get(flow_num, (None, None))
                row[f"{flow_num}OD"] = total
                row[f"Normalized_{flow_num}OD"] = norm
            result_rows.append(row)

        final_df = pd.DataFrame(result_rows, columns=columns)

        suffix_path = Path(suffix)
        output_path = output_root / suffix_path.parent / f"analysis_{suffix_path.name}.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_excel(output_path, index=False, engine="openpyxl")
        output_files.append(output_path)

    return output_files


def copy_reusable_attribute_outputs(
    output_root: Path,
    flow_root: Path,
    suffix_filter: Optional[str] = None,
) -> None:
    if suffix_filter and suffix_filter not in REUSABLE_ATTRIBUTE_SCENARIOS:
        return
    requested = {suffix_filter} if suffix_filter else REUSABLE_ATTRIBUTE_SCENARIOS
    reference_output = output_root / REFERENCE_EXPERIMENT_GROUP
    target_groups = {
        path.name
        for path in flow_root.glob("neighbor_seed_*")
        if path.is_dir() and path.name != REFERENCE_EXPERIMENT_GROUP
    }
    target_groups.update(
        path.name
        for path in output_root.glob("neighbor_seed_*")
        if path.is_dir() and path.name != REFERENCE_EXPERIMENT_GROUP
    )

    if not target_groups:
        return

    for scenario in requested:
        source_analysis = reference_output / f"analysis_{scenario}.xlsx"
        if not source_analysis.exists():
            raise FileNotFoundError(f"Missing static reference carbon analysis: {source_analysis}")

        source_flow_dirs = sorted(reference_output.glob(f"flow*_{scenario}"))
        if not source_flow_dirs:
            raise FileNotFoundError(f"Missing static reference carbon flow directories: {reference_output / ('flow*_' + scenario)}")

        for group in sorted(target_groups):
            target_output = output_root / group
            target_output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_analysis, target_output / source_analysis.name)
            for source_dir in source_flow_dirs:
                target_dir = target_output / source_dir.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(source_dir, target_dir)


def run_pipeline(flow_root: Path, output_root: Path, suffix_filter: Optional[str] = None) -> None:
    groups = discover_flow_dirs(flow_root)
    if suffix_filter:
        groups = {k: v for k, v in groups.items() if k == suffix_filter or Path(k).name == suffix_filter}

    for suffix, flows in sorted(groups.items()):
        for flow_num, flow_dir in sorted(flows.items()):
            print(f"Processing {flow_dir.name}")
            ensure_flow_header(flow_dir)
            updated_dir = output_root / flow_dir.relative_to(flow_root) / "updated"
            compute_emissions(flow_dir, updated_dir)
            normalize_emissions(updated_dir)

    analyze_groups(output_root, groups, flow_root)
    copy_reusable_attribute_outputs(output_root, flow_root, suffix_filter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run carbon emission pipeline for flow outputs.")
    parser.add_argument("--flow-root", type=str, default=None, help="Flow root directory")
    parser.add_argument("--output-root", type=str, default=None, help="Output root directory")
    parser.add_argument("--suffix", type=str, default=None, help="Filter suffix, e.g. modified_density")
    args = parser.parse_args()

    repo_root = get_repo_root()
    flow_root = Path(args.flow_root) if args.flow_root else repo_root / "flow"
    output_root = Path(args.output_root) if args.output_root else repo_root / "outputs" / "emissions"

    run_pipeline(flow_root, output_root, args.suffix)


if __name__ == "__main__":
    main()
