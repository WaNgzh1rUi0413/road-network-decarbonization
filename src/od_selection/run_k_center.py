"""Select ten spatially distributed OD centers with the K-center solver."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.od_selection.k_center_solver import KCenterSolver, load_node_file, save_results
from src.paths import NETWORK_DIR, REPO_ROOT


def parse_city_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    result: set[int] = set()
    for part in raw.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            result.update(range(min(start, end), max(start, end) + 1))
        else:
            result.add(int(part))
    return result


def output_name(node_file: Path, k: int, method: str, revised: bool) -> str:
    stem = node_file.name.removesuffix("_node.csv")
    if not revised and stem.endswith("-original"):
        stem = stem.removesuffix("-original")
    suffix = "_revised" if revised else ""
    return f"{stem}_kcenter_k{k}_{method}{suffix}.csv"


def solve_file(
    node_file: Path,
    output_dir: Path,
    k: int,
    method: str,
    seed: int,
    revised: bool = False,
) -> None:
    points, node_ids = load_node_file(str(node_file))
    solver = KCenterSolver(k=k, random_seed=seed)
    solutions = solver.find_optimal_solutions(
        points, n_runs=200, tolerance=1e-6,
        use_exhaustive=method == "greedy_exhaustive",
        use_milp=method == "milp",
        use_binary_search=method == "binary_search",
    )
    if not solutions:
        raise RuntimeError(f"No K-center solution for {node_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    centers, radius, indices = solutions[0]
    output = output_dir / output_name(node_file, k, method, revised)
    save_results(centers, indices, node_ids, str(output), radius)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "k_center" / "results" / "original")
    parser.add_argument("--city-ids", help="IDs and ranges, for example 1,3,5-10; default is all.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--method", choices=["binary_search", "greedy_random", "greedy_exhaustive", "milp"], default="binary_search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--revised",
        action="store_true",
        help="Process revised '<city>-<scenario>_node.csv' files and use revised output names.",
    )
    args = parser.parse_args()
    selected = parse_city_ids(args.city_ids)
    pattern = "*_node.csv" if args.revised else "*-original_node.csv"
    for node_file in sorted(args.network_dir.glob(pattern)):
        city = int(node_file.name.split("-", 1)[0])
        if selected is not None and city not in selected:
            continue
        expected = args.output_dir / output_name(node_file, args.k, args.method, args.revised)
        if args.skip_existing and expected.exists():
            continue
        solve_file(node_file, args.output_dir, args.k, args.method, args.seed, args.revised)


if __name__ == "__main__":
    main()
