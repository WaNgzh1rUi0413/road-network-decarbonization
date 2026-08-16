from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
NETWORK_DIR = BASE_DIR / "network"
OUTPUT_PATH = BASE_DIR / "outputs" / "network_descriptors.csv"

LINK_COLUMNS = [
    "source",
    "target",
    "weight",
    "capacity",
    "fft",
    "max_speed",
    "constant_1",
    "constant_2",
]

OUTPUT_COLUMNS = [
    "NO",
    "Node Num",
    "Edge Num",
    "Average degree",
    "Clustering coefficient",
    "Transitivity",
    "Pearson degree correlation",
    "Density",
    "Average node strength",
    "Characteristic path length",
    "Efficiency",
]


def read_nodes(path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(path, header=None, names=["node_id", "x", "y"])
    return nodes


def read_links(path: Path) -> pd.DataFrame:
    links = pd.read_csv(path, header=None, names=LINK_COLUMNS)
    return links


def weighted_path_metrics(graph: nx.Graph) -> Tuple[float, float, float]:
    node_count = graph.number_of_nodes()
    if node_count <= 1:
        return 0.0, 0.0, 0.0

    total_distance = 0.0
    total_efficiency = 0.0
    reachable_pairs = 0
    diameter = 0.0

    lengths_by_source: Dict[int, Dict[int, float]] = dict(
        nx.all_pairs_dijkstra_path_length(graph, weight="weight")
    )
    for source, lengths in lengths_by_source.items():
        for target, distance in lengths.items():
            if source == target:
                continue
            reachable_pairs += 1
            total_distance += distance
            diameter = max(diameter, distance)
            if distance > 0:
                total_efficiency += 1 / distance

    average_distance = total_distance / reachable_pairs if reachable_pairs else 0.0
    efficiency = total_efficiency / (node_count * (node_count - 1))
    return diameter, average_distance, efficiency


def calculate_city_metrics(city_id: int) -> dict:
    node_path = NETWORK_DIR / f"{city_id}-original_node.csv"
    link_path = NETWORK_DIR / f"{city_id}-original_link.csv"

    nodes = read_nodes(node_path)
    links = read_links(link_path)

    graph = nx.from_pandas_edgelist(
        links,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.MultiGraph(),
    )
    simple_graph = nx.from_pandas_edgelist(
        links,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.Graph(),
    )

    node_ids = nodes["node_id"].tolist()
    graph.add_nodes_from(node_ids)
    simple_graph.add_nodes_from(node_ids)

    node_num = graph.number_of_nodes()
    edge_num = graph.number_of_edges()
    degrees = dict(nx.degree(graph))
    strengths = dict(nx.degree(graph, weight="weight"))

    average_degree = sum(degrees.values()) / node_num if node_num else 0.0
    average_strength = sum(strengths.values()) / node_num if node_num else 0.0
    diameter, average_distance, efficiency = weighted_path_metrics(simple_graph)
    characteristic_path_length = average_distance / diameter if diameter else 0.0
    normalized_efficiency = efficiency * diameter

    try:
        degree_pearson = nx.degree_pearson_correlation_coefficient(graph)
    except Exception:
        degree_pearson = 0.0

    return {
        "NO": city_id,
        "Node Num": node_num,
        "Edge Num": edge_num,
        "Average degree": average_degree,
        "Clustering coefficient": nx.average_clustering(simple_graph, weight="weight"),
        "Transitivity": nx.transitivity(simple_graph),
        "Pearson degree correlation": degree_pearson,
        "Density": nx.density(graph),
        "Average node strength": average_strength,
        "Characteristic path length": characteristic_path_length,
        "Efficiency": normalized_efficiency,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []
    for city_id in range(1, 141):
        node_path = NETWORK_DIR / f"{city_id}-original_node.csv"
        link_path = NETWORK_DIR / f"{city_id}-original_link.csv"
        if not node_path.exists() or not link_path.exists():
            print(f"Skip city {city_id}: missing node/link file")
            continue

        try:
            print(f"Calculating indicators for city {city_id}")
            results.append(calculate_city_metrics(city_id))
        except Exception as exc:
            print(f"Error processing city {city_id}: {exc}")

    results_df = pd.DataFrame(results).sort_values("NO")
    results_df = results_df.reindex(columns=OUTPUT_COLUMNS)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(results_df)} city indicators to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
