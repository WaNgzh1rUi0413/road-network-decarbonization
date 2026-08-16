import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parents[2])
NETWORK_DIR = os.path.join(BASE_DIR, "network")
OUTPUT_DIR = os.path.join(BASE_DIR, "revised_network")

METRIC_TOL = 1e-3


@dataclass
class ValidationResult:
    city_id: int
    scenario: str
    experiment_group: str
    density_old: float
    density_new: float
    avg_degree_old: float
    avg_degree_new: float
    density_unchanged: bool
    avg_degree_unchanged: bool
    node_count_unchanged: bool
    edge_count_unchanged: bool
    capacity_unchanged_on_orig: bool
    speed_unchanged_on_orig: bool
    target_changed: bool
    expected_pass: bool
    expected_edge_count_match: bool
    rounding_consistent: bool
    edge_rounding_diff: float
    mismatch_note: str


def _read_nodes(path: str) -> pd.DataFrame:
    nodes = pd.read_csv(path, header=None)
    nodes.columns = ["node_id", "x", "y"]
    return nodes


def _read_links(path: str) -> pd.DataFrame:
    links = pd.read_csv(path, header=None)
    links.columns = [
        "from",
        "to",
        "length",
        "capacity",
        "free_flow_time",
        "speed_limit",
        "attr6",
        "attr7",
    ]
    return links


def _density(n: int, e: int) -> float:
    return 2 * e / (n * (n - 1)) if n > 1 else 0.0


def _avg_degree(n: int, e: int) -> float:
    return 2 * e / n if n > 0 else 0.0


def _iter_outputs() -> List[Tuple[int, str, str, str, str]]:
    entries = []
    for experiment_group in os.listdir(OUTPUT_DIR):
        experiment_base = os.path.join(OUTPUT_DIR, experiment_group)
        if not experiment_group.startswith("neighbor_seed_") or not os.path.isdir(experiment_base):
            continue
        for scenario_dir in [
            "modified_density",
            "modified_degree",
            "modified_capacity",
            "modified_speed_limit",
        ]:
            base = os.path.join(experiment_base, scenario_dir)
            if not os.path.isdir(base):
                continue
            for fname in os.listdir(base):
                if not fname.endswith("_link.csv"):
                    continue
                parts = fname.split("-")
                if not parts:
                    continue
                try:
                    city_id = int(parts[0])
                except ValueError:
                    continue
                link_path = os.path.join(base, fname)
                node_path = link_path.replace("_link.csv", "_node.csv")
                if not os.path.exists(node_path):
                    continue
                entries.append((city_id, scenario_dir, experiment_group, node_path, link_path))
    return entries


def _compare_value_multiset(
    orig_links: pd.DataFrame,
    new_links: pd.DataFrame,
    col: str,
    allow_missing: bool,
) -> Tuple[bool, int]:
    orig = orig_links[["from", "to", col]].copy()
    new = new_links[["from", "to", col]].copy()

    orig_groups = orig.groupby(["from", "to"])[col].apply(list).to_dict()
    new_groups = new.groupby(["from", "to"])[col].apply(list).to_dict()

    mismatches = 0
    for key, orig_vals in orig_groups.items():
        if key not in new_groups:
            if not allow_missing:
                mismatches += len(orig_vals)
            continue
        new_vals = new_groups[key]
        if not allow_missing:
            if len(orig_vals) != len(new_vals):
                mismatches += abs(len(orig_vals) - len(new_vals))
                continue
            orig_arr = np.array(sorted(orig_vals))
            new_arr = np.array(sorted(new_vals))
            if np.issubdtype(orig_arr.dtype, np.number) and np.issubdtype(new_arr.dtype, np.number):
                comp = np.isclose(orig_arr, new_arr, rtol=1e-6, atol=1e-6, equal_nan=False)
                mismatches += int((~comp).sum())
            else:
                mismatches += int((orig_arr != new_arr).sum())
            continue

        if len(new_vals) > len(orig_vals):
            mismatches += len(new_vals) - len(orig_vals)
            continue

        orig_used = [False] * len(orig_vals)
        if np.issubdtype(np.array(orig_vals).dtype, np.number):
            for nv in new_vals:
                found = False
                for i, ov in enumerate(orig_vals):
                    if orig_used[i]:
                        continue
                    if np.isclose(float(nv), float(ov), rtol=1e-6, atol=1e-6, equal_nan=False):
                        orig_used[i] = True
                        found = True
                        break
                if not found:
                    mismatches += 1
        else:
            for nv in new_vals:
                found = False
                for i, ov in enumerate(orig_vals):
                    if orig_used[i]:
                        continue
                    if nv == ov:
                        orig_used[i] = True
                        found = True
                        break
                if not found:
                    mismatches += 1

    return mismatches == 0, int(mismatches)


def main() -> None:
    results: List[ValidationResult] = []
    for city_id, scenario_dir, experiment_group, node_path, link_path in _iter_outputs():
        orig_node = os.path.join(NETWORK_DIR, f"{city_id}-original_node.csv")
        orig_link = os.path.join(NETWORK_DIR, f"{city_id}-original_link.csv")
        if not os.path.exists(orig_node) or not os.path.exists(orig_link):
            continue

        nodes_old = _read_nodes(orig_node)
        links_old = _read_links(orig_link)
        nodes_new = _read_nodes(node_path)
        links_new = _read_links(link_path)

        n_old = len(nodes_old)
        e_old = len(links_old)
        n_new = len(nodes_new)
        e_new = len(links_new)

        density_old = _density(n_old, e_old)
        density_new = _density(n_new, e_new)
        avg_degree_old = _avg_degree(n_old, e_old)
        avg_degree_new = _avg_degree(n_new, e_new)

        density_unchanged = abs(density_new - density_old) <= METRIC_TOL
        avg_degree_unchanged = abs(avg_degree_new - avg_degree_old) <= METRIC_TOL
        node_count_unchanged = n_old == n_new
        edge_count_unchanged = e_old == e_new

        allow_missing = scenario_dir in {"modified_density", "modified_degree"}
        capacity_unchanged_on_orig, cap_mismatch = _compare_value_multiset(
            links_old,
            links_new,
            "capacity",
            allow_missing,
        )
        speed_unchanged_on_orig, speed_mismatch = _compare_value_multiset(
            links_old,
            links_new,
            "speed_limit",
            allow_missing,
        )

        target_changed = False
        expected_edge_count_match = True
        rounding_consistent = True
        edge_rounding_diff = 0.0
        mismatch_note = ""
        if scenario_dir == "modified_density":
            expected_e = avg_degree_old * n_new / 2.0
            edge_rounding_diff = abs(e_new - expected_e)
            expected_edge_count_match = edge_rounding_diff <= 0.5
            avg_expected = 2 * int(round(expected_e)) / n_new if n_new > 0 else 0.0
            rounding_consistent = abs(avg_degree_new - avg_expected) <= METRIC_TOL
            target_changed = not density_unchanged
            expected_pass = (
                avg_degree_unchanged
                and capacity_unchanged_on_orig
                and speed_unchanged_on_orig
            )
        elif scenario_dir == "modified_degree":
            expected_e = density_old * n_new * (n_new - 1) / 2.0
            edge_rounding_diff = abs(e_new - expected_e)
            expected_edge_count_match = edge_rounding_diff <= 0.5
            density_expected = _density(n_new, int(round(expected_e)))
            rounding_consistent = abs(density_new - density_expected) <= METRIC_TOL
            target_changed = not avg_degree_unchanged
            expected_pass = (
                density_unchanged
                and capacity_unchanged_on_orig
                and speed_unchanged_on_orig
            )
        elif scenario_dir == "modified_capacity":
            target_changed = True
            expected_pass = (
                density_unchanged
                and avg_degree_unchanged
                and speed_unchanged_on_orig
            )
        else:
            target_changed = True
            expected_pass = (
                density_unchanged
                and avg_degree_unchanged
                and capacity_unchanged_on_orig
            )

        if cap_mismatch:
            mismatch_note += f"cap_cols_mismatch={cap_mismatch};"
        if speed_mismatch:
            mismatch_note += f"speed_cols_mismatch={speed_mismatch};"

        results.append(
            ValidationResult(
                city_id=city_id,
                scenario=scenario_dir,
                experiment_group=experiment_group,
                density_old=density_old,
                density_new=density_new,
                avg_degree_old=avg_degree_old,
                avg_degree_new=avg_degree_new,
                density_unchanged=density_unchanged,
                avg_degree_unchanged=avg_degree_unchanged,
                node_count_unchanged=node_count_unchanged,
                edge_count_unchanged=edge_count_unchanged,
                capacity_unchanged_on_orig=capacity_unchanged_on_orig,
                speed_unchanged_on_orig=speed_unchanged_on_orig,
                target_changed=target_changed,
                expected_pass=expected_pass,
                expected_edge_count_match=expected_edge_count_match,
                rounding_consistent=rounding_consistent,
                edge_rounding_diff=edge_rounding_diff,
                mismatch_note=mismatch_note,
            )
        )

    if not results:
        print("No outputs found to validate.")
        return

    df = pd.DataFrame([r.__dict__ for r in results])
    out_path = os.path.join(OUTPUT_DIR, "validation_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Validation summary saved: {out_path}")

    grouped = df.groupby("city_id")
    for city_id, group in grouped:
        print(f"City {city_id}:")
        for _, row in group.iterrows():
            scenario = row["scenario"]
            experiment_group = row["experiment_group"]
            if scenario == "modified_density":
                ok = (
                    row["avg_degree_unchanged"]
                    and row["capacity_unchanged_on_orig"]
                    and row["speed_unchanged_on_orig"]
                )
                msg = (
                    f"  {experiment_group}/{scenario}: avg_degree={row['avg_degree_unchanged']}, "
                    f"capacity={row['capacity_unchanged_on_orig']}, "
                    f"speed={row['speed_unchanged_on_orig']}, "
                    f"edge_round={row['expected_edge_count_match']}, "
                    f"edge_diff={row['edge_rounding_diff']:.3f}, "
                    f"rounding_ok={row['rounding_consistent']} -> {ok}"
                )
            elif scenario == "modified_degree":
                ok = (
                    row["density_unchanged"]
                    and row["capacity_unchanged_on_orig"]
                    and row["speed_unchanged_on_orig"]
                )
                msg = (
                    f"  {experiment_group}/{scenario}: density={row['density_unchanged']}, "
                    f"capacity={row['capacity_unchanged_on_orig']}, "
                    f"speed={row['speed_unchanged_on_orig']}, "
                    f"edge_round={row['expected_edge_count_match']}, "
                    f"edge_diff={row['edge_rounding_diff']:.3f}, "
                    f"rounding_ok={row['rounding_consistent']} -> {ok}"
                )
            elif scenario == "modified_capacity":
                ok = (
                    row["density_unchanged"]
                    and row["avg_degree_unchanged"]
                    and row["speed_unchanged_on_orig"]
                )
                msg = (
                    f"  {experiment_group}/{scenario}: density={row['density_unchanged']}, "
                    f"avg_degree={row['avg_degree_unchanged']}, "
                    f"speed={row['speed_unchanged_on_orig']} -> {ok}"
                )
            else:
                ok = (
                    row["density_unchanged"]
                    and row["avg_degree_unchanged"]
                    and row["capacity_unchanged_on_orig"]
                )
                msg = (
                    f"  {experiment_group}/{scenario}: density={row['density_unchanged']}, "
                    f"avg_degree={row['avg_degree_unchanged']}, "
                    f"capacity={row['capacity_unchanged_on_orig']} -> {ok}"
                )
            print(msg)


if __name__ == "__main__":
    main()
