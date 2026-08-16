"""Download and standardize the 140 OpenStreetMap road networks.

This optional upstream step depends on the live OpenStreetMap service. The
versioned node/link tables under ``network/`` are the exact inputs used for the
reported experiments and should be used for strict numerical reproduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

from src.paths import NETWORK_DIR, REPO_ROOT


def tolerance_for_city(city_id: int) -> int:
    if city_id == 88:
        return 20
    special = {31, 38, 39, 40, 42, 43, 44, 46, 52, 54, 59, 64, 78, 79}
    return 15 if city_id >= 91 or city_id in special else 30


def capacity_and_speed(lanes: int) -> tuple[float, float]:
    if lanes == 1:
        return 1300.0, 11.11
    if lanes == 2:
        return 2800.0, 16.67
    return 1750.0 * lanes, 22.22


def build_city(city_id: int, polygon_points: list[list[float]], output_dir: Path) -> None:
    graph = ox.graph_from_polygon(Polygon(polygon_points), network_type="drive", simplify=True)
    projected = ox.project_graph(graph)
    consolidated = ox.consolidate_intersections(
        projected, rebuild_graph=True, tolerance=tolerance_for_city(city_id), dead_ends=False
    )
    undirected = nx.MultiGraph(consolidated).to_undirected()
    nodes, _ = ox.graph_to_gdfs(undirected, nodes=True, edges=True)
    node_table = nodes[["x", "y"]].reset_index()
    node_table.columns = ["node_id", "x", "y"]

    raw = []
    for u, v, attrs in undirected.edges(data=True):
        lanes = attrs.get("lanes")
        if isinstance(lanes, list):
            lanes = lanes[0]
        try:
            lanes = int(lanes)
        except (TypeError, ValueError):
            lanes = np.nan
        raw.append((u, v, float(attrs.get("length", 0)), lanes))
    raw_table = pd.DataFrame(raw, columns=["source", "target", "length", "lanes"])
    fill = int(raw_table["lanes"].mode().iloc[0]) if raw_table["lanes"].notna().any() else 2
    raw_table["lanes"] = raw_table["lanes"].fillna(fill).astype(int)
    edges = []
    for row in raw_table.itertuples(index=False):
        capacity, speed = capacity_and_speed(row.lanes)
        edges.append((row.source, row.target, row.length, capacity, row.length / speed / 3600, speed, 0.15, 4))

    output_dir.mkdir(parents=True, exist_ok=True)
    node_table.to_csv(output_dir / f"{city_id}-original_node.csv", index=False, header=False)
    pd.DataFrame(edges).to_csv(output_dir / f"{city_id}-original_link.csv", index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "paper_config.json")
    parser.add_argument("--city-ids", help="Comma-separated city IDs; default is all 140 cities.")
    parser.add_argument("--output-dir", type=Path, default=NETWORK_DIR)
    args = parser.parse_args()
    polygons = json.loads(args.config.read_text(encoding="utf-8"))["polygon_data"]
    selected = set(range(1, len(polygons) + 1))
    if args.city_ids:
        selected = {int(v) for v in args.city_ids.split(",")}
    for city_id, polygon in enumerate(polygons, start=1):
        if city_id in selected:
            print(f"building city {city_id}")
            build_city(city_id, polygon, args.output_dir)


if __name__ == "__main__":
    main()

