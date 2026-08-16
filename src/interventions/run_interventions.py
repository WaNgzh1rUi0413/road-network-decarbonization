import argparse
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from src.interventions.network_builder import build_congested_edge_network, build_realistic_network
from src.interventions.random_seeds import NEIGHBOR_EXPERIMENTS, NODE_OFFSET_FACTOR, derive_random_seed

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))

BASE_DIR = str(Path(__file__).resolve().parents[2])
NETWORK_DIR = os.path.join(BASE_DIR, "network")
FLOW_ROOT_DIR = os.path.join(BASE_DIR, "flow")
FLOW_DIR = os.path.join(FLOW_ROOT_DIR, "flow10")
OUTPUT_DIR = os.path.join(BASE_DIR, "revised_network")

TOP_CONGESTED_PERCENT = 0.20
CAPACITY_MULTIPLIER = 1.20
SPEED_LIMIT_MULTIPLIER = 1.20
TARGET_METRIC_CHANGE = 0.20  # used by `proportional` strategy: network-level change
TARGET_TOLERANCE = 1e-6
MAX_EXTRA_PER_NEW = 3
MAX_DEGREE = 5
ALLOW_PARTIAL_DEGREE = False
RANKING_OD = 10
REFERENCE_EXPERIMENT_GROUP = NEIGHBOR_EXPERIMENTS[0][0]
REUSABLE_ATTRIBUTE_SCENARIOS = {"modified_capacity", "modified_speed_limit"}
REVISED_NETWORK_DIR = os.path.join(BASE_DIR, "revised_network")


def parse_change_ratio(value: float) -> float:
    ratio = float(value)
    if ratio < 0:
        raise ValueError("--change-ratio must be non-negative")
    if ratio > 1:
        ratio = ratio / 100.0
    return ratio


def configure_change_ratio(value: float) -> None:
    global TARGET_METRIC_CHANGE, CAPACITY_MULTIPLIER, SPEED_LIMIT_MULTIPLIER
    TARGET_METRIC_CHANGE = parse_change_ratio(value)
    CAPACITY_MULTIPLIER = 1.0 + TARGET_METRIC_CHANGE
    SPEED_LIMIT_MULTIPLIER = 1.0 + TARGET_METRIC_CHANGE


class SkippedRevision(Exception):
    def __init__(self, city_id: int, scenario: str, experiment_group: str, reason: str):
        super().__init__(reason)
        self.city_id = city_id
        self.scenario = scenario
        self.experiment_group = experiment_group
        self.reason = reason


def recalculate_free_flow_time(length: pd.Series, speed_limit: pd.Series, fallback: pd.Series) -> pd.Series:
    valid_speed = speed_limit > 0
    updated = fallback.copy()
    updated.loc[valid_speed] = length.loc[valid_speed] / speed_limit.loc[valid_speed] / 3600.0
    return updated


def apply_mean_attributes_to_added_edges(
    original_nodes: pd.DataFrame,
    original_links: pd.DataFrame,
    revised_links: pd.DataFrame,
) -> pd.DataFrame:
    original_node_ids = set(original_nodes["node_id"].astype(float))
    added_mask = ~(
        revised_links["from"].astype(float).isin(original_node_ids)
        & revised_links["to"].astype(float).isin(original_node_ids)
    )
    if not added_mask.any():
        return revised_links

    revised_links = revised_links.copy()
    mean_capacity = float(original_links["capacity"].mean())
    mean_speed_limit = float(original_links["speed_limit"].mean())
    mean_attr6 = float(original_links["attr6"].mean())
    mean_attr7 = float(original_links["attr7"].mean())
    mean_free_flow_time = float(original_links["free_flow_time"].mean())

    revised_links.loc[added_mask, "capacity"] = mean_capacity
    revised_links.loc[added_mask, "speed_limit"] = mean_speed_limit
    revised_links.loc[added_mask, "attr6"] = mean_attr6
    revised_links.loc[added_mask, "attr7"] = mean_attr7
    revised_links.loc[added_mask, "free_flow_time"] = recalculate_free_flow_time(
        revised_links.loc[added_mask, "length"],
        revised_links.loc[added_mask, "speed_limit"],
        pd.Series(mean_free_flow_time, index=revised_links.index[added_mask]),
    )
    return revised_links


@dataclass
class EdgeTemplate:
    length: float
    capacity: float
    free_flow_time: float
    speed_limit: float
    attr6: float
    attr7: float


@dataclass
class RevisionResult:
    city_id: int
    scenario: str
    experiment_group: str
    neighbor_seed: int
    n_old: int
    e_old: int
    n_new: int
    e_new: int
    delta_n: int
    delta_e: int
    delta_e_actual: int
    density_old: float
    density_new: float
    avg_degree_old: float
    avg_degree_new: float


@dataclass
class RevisionOutput:
    result: RevisionResult
    original_node: str
    original_link: str
    revised_node: str
    revised_link: str
    scenario: str
    city_id: int
    experiment_group: str


def read_nodes(path: str) -> pd.DataFrame:
    nodes = pd.read_csv(path, header=None)
    nodes.columns = ["node_id", "x", "y"]
    return nodes


def read_links(path: str) -> pd.DataFrame:
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


def scenario_output_paths(city_id: int, scenario: str, experiment_group: str) -> Tuple[str, str]:
    out_dir = os.path.join(OUTPUT_DIR, experiment_group)
    node_name = f"{city_id}-{scenario}_node.csv"
    link_name = f"{city_id}-{scenario}_link.csv"
    return os.path.join(out_dir, node_name), os.path.join(out_dir, link_name)


def remove_revision_files(city_id: int, scenario: str, experiment_group: str) -> None:
    node_path, link_path = scenario_output_paths(city_id, scenario, experiment_group)
    revised_node = os.path.join(REVISED_NETWORK_DIR, experiment_group, f"{city_id}-{scenario}_node.csv")
    revised_link = os.path.join(REVISED_NETWORK_DIR, experiment_group, f"{city_id}-{scenario}_link.csv")
    ta_dir = os.path.join(REVISED_NETWORK_DIR, experiment_group, "traffic_assignment_input")
    ta_base = os.path.join(ta_dir, f"{city_id}-{scenario}new")
    paths = [
        node_path,
        link_path,
        revised_node,
        revised_link,
        f"{ta_base}_node.csv",
        f"{ta_base}_link.csv",
        f"{ta_base}_od.csv",
    ]
    for od in range(1, 11):
        paths.append(
            os.path.join(
                BASE_DIR,
                "flow",
                experiment_group,
                f"flow{od}_{scenario}",
                f"{city_id}-{scenario}_flow.csv",
            )
        )
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def build_revision_output_from_files(
    city_id: int,
    scenario: str,
    experiment_group: str,
    neighbor_seed: int,
    original_node: str,
    original_link: str,
    revised_node: str,
    revised_link: str,
) -> RevisionOutput:
    nodes = read_nodes(original_node)
    links = read_links(original_link)
    revised_nodes = read_nodes(revised_node)
    revised_links = read_links(revised_link)

    n_old = len(nodes)
    e_old = len(links)
    n_new = len(revised_nodes)
    e_new = len(revised_links)
    density_old = 2 * e_old / (n_old * (n_old - 1)) if n_old > 1 else 0.0
    density_new = 2 * e_new / (n_new * (n_new - 1)) if n_new > 1 else 0.0
    avg_degree_old = 2 * e_old / n_old if n_old > 0 else 0.0
    avg_degree_new = 2 * e_new / n_new if n_new > 0 else 0.0

    result = RevisionResult(
        city_id=city_id,
        scenario=scenario,
        experiment_group=experiment_group,
        neighbor_seed=neighbor_seed,
        n_old=n_old,
        e_old=e_old,
        n_new=n_new,
        e_new=e_new,
        delta_n=n_new - n_old,
        delta_e=e_new - e_old,
        delta_e_actual=e_new - e_old,
        density_old=density_old,
        density_new=density_new,
        avg_degree_old=avg_degree_old,
        avg_degree_new=avg_degree_new,
    )

    return RevisionOutput(
        result=result,
        original_node=original_node,
        original_link=original_link,
        revised_node=revised_node,
        revised_link=revised_link,
        scenario=scenario,
        city_id=city_id,
        experiment_group=experiment_group,
    )


def reuse_reference_attribute_revision(
    city_id: int,
    scenario: str,
    experiment_group: str,
    neighbor_seed: int,
) -> RevisionOutput | None:
    if scenario not in REUSABLE_ATTRIBUTE_SCENARIOS or experiment_group == REFERENCE_EXPERIMENT_GROUP:
        return None

    ref_node, ref_link = scenario_output_paths(city_id, scenario, REFERENCE_EXPERIMENT_GROUP)
    if not (os.path.exists(ref_node) and os.path.exists(ref_link)):
        return None

    out_node, out_link = scenario_output_paths(city_id, scenario, experiment_group)
    os.makedirs(os.path.dirname(out_node), exist_ok=True)
    shutil.copy2(ref_node, out_node)
    shutil.copy2(ref_link, out_link)

    original_node = os.path.join(NETWORK_DIR, f"{city_id}-original_node.csv")
    original_link = os.path.join(NETWORK_DIR, f"{city_id}-original_link.csv")
    return build_revision_output_from_files(
        city_id,
        scenario,
        experiment_group,
        neighbor_seed,
        original_node,
        original_link,
        out_node,
        out_link,
    )


def read_flow(path: str) -> pd.DataFrame:
    # First two lines are metadata like AvePathNum and ConLinkPer
    flow = pd.read_csv(path, header=None, skiprows=2)
    flow.columns = [
        "from",
        "to",
        "length",
        "capacity",
        "free_flow_time",
        "speed_limit",
        "attr6",
        "attr7",
        "flow",
        "time",
        "speed",
    ]
    return flow


def find_flow_path(city_id: int) -> str:
    od_dir = f"flow{RANKING_OD}"
    candidates = [
        os.path.join(FLOW_ROOT_DIR, "original", od_dir, f"{city_id}-original_flow.csv"),
        os.path.join(FLOW_ROOT_DIR, "original", od_dir, f"{city_id}_flow.csv"),
        os.path.join(FLOW_ROOT_DIR, od_dir, f"{city_id}-original_flow.csv"),
        os.path.join(FLOW_ROOT_DIR, od_dir, f"{city_id}_flow.csv"),
    ]
    if RANKING_OD == 10:
        candidates.extend(
            [
                os.path.join(FLOW_DIR, f"{city_id}-original_flow.csv"),
                os.path.join(FLOW_DIR, f"{city_id}_flow.csv"),
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def build_template_map(flow: pd.DataFrame, fallback: EdgeTemplate) -> Dict[int, EdgeTemplate]:
    flow = flow.copy()
    flow["ratio"] = flow["flow"] / flow["capacity"].replace(0, np.nan)
    flow["ratio"] = flow["ratio"].fillna(0)

    templates: Dict[int, EdgeTemplate] = {}
    for node_col in ["from", "to"]:
        grouped = flow.sort_values("ratio", ascending=False).groupby(node_col)
        for node_id, group in grouped:
            if node_id in templates:
                continue
            row = group.iloc[0]
            templates[int(node_id)] = EdgeTemplate(
                length=float(row["length"]),
                capacity=float(row["capacity"]),
                free_flow_time=float(row["free_flow_time"]),
                speed_limit=float(row["speed_limit"]),
                attr6=float(row["attr6"]),
                attr7=float(row["attr7"]),
            )
    return templates


def compute_node_congestion(flow: pd.DataFrame) -> pd.Series:
    flow = flow.copy()
    flow["ratio"] = flow["flow"] / flow["capacity"].replace(0, np.nan)
    flow["ratio"] = flow["ratio"].fillna(0)

    from_scores = flow[["from", "ratio"]].rename(columns={"from": "node"})
    to_scores = flow[["to", "ratio"]].rename(columns={"to": "node"})
    all_scores = pd.concat([from_scores, to_scores], axis=0, ignore_index=True)
    return all_scores.groupby("node")["ratio"].mean()


def pick_top_nodes(node_scores: pd.Series, nodes: Iterable[int], percent: float) -> List[int]:
    node_scores = node_scores.reindex(list(nodes)).fillna(0)
    top_n = max(1, math.ceil(len(node_scores) * percent))
    return node_scores.sort_values(ascending=False).head(top_n).index.astype(int).tolist()



def calc_delta_e_density(density_old: float, n_old: int, e_old: int, delta_n: int) -> int:
    n_new = n_old + delta_n
    e_needed = density_old * n_new * (n_new - 1) / 2
    return int(round(e_needed - e_old))


def calc_delta_e_avg_degree(avg_degree_old: float, delta_n: int) -> int:
    e_needed = avg_degree_old * delta_n / 2
    return int(round(e_needed))


def make_new_nodes(
    nodes: pd.DataFrame,
    base_nodes: List[int],
    delta_n: int,
) -> Tuple[pd.DataFrame, List[int], Dict[int, int]]:
    node_lookup = nodes.set_index("node_id")[["x", "y"]]
    max_id = int(nodes["node_id"].max())

    new_rows = []
    new_ids: List[int] = []
    base_map: Dict[int, int] = {}

    for i in range(delta_n):
        base = base_nodes[i % len(base_nodes)]
        base_x, base_y = node_lookup.loc[base]
        new_id = max_id + 1 + i
        offset = 5.0 + (i % 5) * 0.5
        new_rows.append([new_id, base_x + offset, base_y + offset])
        new_ids.append(new_id)
        base_map[new_id] = base

    new_nodes = pd.DataFrame(new_rows, columns=["node_id", "x", "y"])
    return new_nodes, new_ids, base_map


def build_edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def build_ratio_map(flow: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    flow = flow.copy()
    flow["ratio"] = flow["flow"] / flow["capacity"].replace(0, np.nan)
    flow["ratio"] = flow["ratio"].fillna(0.0)
    ratio_map: Dict[Tuple[int, int], float] = {}
    for a, b, r in flow[["from", "to", "ratio"]].values:
        ra = float(r)
        ratio_map[(int(a), int(b))] = ra
        ratio_map[(int(b), int(a))] = ra
    return ratio_map


def build_edge_ratio_list(flow: pd.DataFrame, links: pd.DataFrame) -> List[Tuple[int, int, float]]:
    flow = flow.copy()
    ratio_map = build_ratio_map(flow)
    seen = set()
    edge_list = []
    for u, v in links[["from", "to"]].values:
        u = int(u)
        v = int(v)
        key = build_edge_key(u, v)
        if key in seen:
            continue
        seen.add(key)
        ratio = ratio_map.get((u, v), ratio_map.get((v, u), 0.0))
        edge_list.append((u, v, float(ratio)))
    edge_list.sort(key=lambda x: x[2], reverse=True)
    return edge_list


def apply_ordered_metric_boost(
    links: pd.DataFrame,
    flow: pd.DataFrame,
    attr: str,
    max_multiplier: float,
) -> None:
    ratio_map = build_ratio_map(flow)
    vc_series = links.apply(
        lambda r: ratio_map.get((int(r["from"]), int(r["to"])), ratio_map.get((int(r["to"]), int(r["from"])), 0.0)),
        axis=1,
    ).astype(float)
    metric = links[attr].astype(float)
    target_increase = float(metric.sum()) * TARGET_METRIC_CHANGE
    remaining = target_increase
    multiplier = pd.Series(1.0, index=links.index)

    for idx in vc_series.sort_values(ascending=False, kind="mergesort").index:
        if remaining <= 1e-9:
            break
        base_value = float(metric.loc[idx])
        if not np.isfinite(base_value) or base_value <= 0:
            continue
        max_delta = base_value * (max_multiplier - 1.0)
        if max_delta <= 0:
            continue
        delta = min(max_delta, remaining)
        multiplier.loc[idx] = 1.0 + delta / base_value
        remaining -= delta

    links[attr] = metric * multiplier


def pick_top_edges(
    flow: pd.DataFrame,
    links: pd.DataFrame,
    percent: float,
) -> List[Tuple[int, int, float]]:
    edge_list = build_edge_ratio_list(flow, links)
    if not edge_list:
        return []
    top_n = max(1, math.ceil(len(edge_list) * percent))
    return edge_list[:top_n]


def make_new_edges(
    links: pd.DataFrame,
    new_ids: List[int],
    base_map: Dict[int, int],
    base_nodes: List[int],
    delta_e: int,
    templates: Dict[int, EdgeTemplate],
    fallback: EdgeTemplate,
) -> pd.DataFrame:
    existing = set(build_edge_key(int(a), int(b)) for a, b in links[["from", "to"]].values)

    new_rows = []
    used_edges = set()

    def add_edge(u: int, v: int) -> bool:
        key = build_edge_key(u, v)
        if key in existing or key in used_edges or u == v:
            return False
        template = templates.get(v, templates.get(u, fallback))
        new_rows.append(
            [
                u,
                v,
                template.length,
                template.capacity,
                template.free_flow_time,
                template.speed_limit,
                template.attr6,
                template.attr7,
            ]
        )
        used_edges.add(key)
        return True

    # First, connect each new node to its base congested node
    for new_id in new_ids:
        if len(new_rows) >= delta_e:
            break
        add_edge(new_id, base_map[new_id])

    # Add remaining edges by connecting new nodes to top congested nodes
    if len(new_rows) < delta_e:
        attempts = 0
        max_attempts = delta_e * 20
        i = 0
        while len(new_rows) < delta_e and attempts < max_attempts:
            new_id = new_ids[i % len(new_ids)]
            target = base_nodes[(i // len(new_ids)) % len(base_nodes)]
            add_edge(new_id, target)
            i += 1
            attempts += 1

    return pd.DataFrame(
        new_rows,
        columns=[
            "from",
            "to",
            "length",
            "capacity",
            "free_flow_time",
            "speed_limit",
            "attr6",
            "attr7",
        ],
    )


def calc_max_density_preserving_nodes(density_old: float, n_old: int, max_degree: int) -> int:
    if density_old <= 0 or max_degree <= 0:
        return 10**9
    max_n_new = int(math.floor(1.0 + max_degree / density_old))
    return max(0, max_n_new - n_old)


def revise_city(
    city_id: int,
    scenario: str,
    revision_strategy: str = "edge_count",
    experiment_group: str = "neighbor_seed_01",
    neighbor_seed: int = 101,
) -> RevisionOutput:
    node_path = os.path.join(NETWORK_DIR, f"{city_id}-original_node.csv")
    link_path = os.path.join(NETWORK_DIR, f"{city_id}-original_link.csv")
    flow_path = find_flow_path(city_id)

    nodes = read_nodes(node_path)
    links = read_links(link_path)
    flow = read_flow(flow_path)

    n_old = len(nodes)
    e_old = len(links)
    density_old = 2 * e_old / (n_old * (n_old - 1)) if n_old > 1 else 0.0
    avg_degree_old = 2 * e_old / n_old if n_old > 0 else 0.0
    random_seed = derive_random_seed(neighbor_seed, city_id, scenario)
    if scenario in {"modified_capacity", "modified_speed_limit"}:
        delta_n = 0
        delta_e = 0
        all_nodes = nodes.copy()
        all_links = links.copy()

        if scenario == "modified_capacity":
            all_links["capacity"] = all_links["capacity"] * CAPACITY_MULTIPLIER
        else:
            all_links["speed_limit"] = all_links["speed_limit"] * SPEED_LIMIT_MULTIPLIER
            all_links["free_flow_time"] = recalculate_free_flow_time(
                all_links["length"],
                all_links["speed_limit"],
                all_links["free_flow_time"],
            )
    else:
        candidates = build_edge_ratio_list(flow, links)
        if revision_strategy == "proportional":
            if scenario == "modified_density":
                # Closed-form: density_new / density_old = (n_old-1)/(n_old+dn-1) = 1 - TARGET
                # ⇒ dn = (n_old - 1) × TARGET / (1 - TARGET)
                delta_n = int(round((n_old - 1) * TARGET_METRIC_CHANGE / (1.0 - TARGET_METRIC_CHANGE)))
                delta_e = calc_delta_e_avg_degree(avg_degree_old, delta_n)
            elif scenario == "modified_degree":
                # avg_deg_new / avg_deg_old = (n_old+dn-1)/(n_old-1) = 1 + TARGET
                # ⇒ dn = (n_old - 1) × TARGET. delta_e then enforced by
                # density-preserving formula; the builder's soft-relaxation
                # (max_degree 4 → 5) fills the gap that strict 4 cannot.
                delta_n = int(round((n_old - 1) * TARGET_METRIC_CHANGE))
                delta_n = max(0, min(delta_n, len(candidates)))
                delta_e = calc_delta_e_density(density_old, n_old, e_old, delta_n)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
        else:
            target_count = max(1, math.ceil(len(candidates) * TOP_CONGESTED_PERCENT)) if candidates else 0
            delta_n = target_count

            if scenario == "modified_density":
                # Keep average degree fixed so density changes with N.
                delta_e = calc_delta_e_avg_degree(avg_degree_old, delta_n)
            elif scenario == "modified_degree":
                # Keep density fixed so average degree changes with N.
                max_delta_n = calc_max_density_preserving_nodes(density_old, n_old, MAX_DEGREE)
                delta_n = max(0, min(delta_n, max_delta_n))
                delta_e = calc_delta_e_density(density_old, n_old, e_old, delta_n)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

        if delta_e < 0:
            delta_e = 0

        fallback = EdgeTemplate(
            length=float(links["length"].median()),
            capacity=float(links["capacity"].median()),
            free_flow_time=float(links["free_flow_time"].median()),
            speed_limit=float(links["speed_limit"].median()),
            attr6=float(links["attr6"].median()),
            attr7=float(links["attr7"].median()),
        )
        templates = build_template_map(flow, fallback)

        if not candidates or delta_n == 0:
            all_nodes = nodes.copy()
            all_links = links.copy()
            edges_added = 0
        elif revision_strategy in {"congested_edges", "proportional"}:
            all_nodes, all_links, edges_added, actual_delta_n = build_congested_edge_network(
                nodes,
                links,
                candidates,
                delta_n,
                delta_e,
                templates,
                fallback,
                node_offset_factor=NODE_OFFSET_FACTOR,
                max_extra_per_new=MAX_EXTRA_PER_NEW,
                max_degree=MAX_DEGREE,
                random_seed=random_seed,
                relaxed_max_degree=0,
            )
            delta_n = actual_delta_n
            if scenario == "modified_density":
                delta_e = calc_delta_e_avg_degree(avg_degree_old, delta_n)
            elif scenario == "modified_degree":
                delta_e = calc_delta_e_density(density_old, n_old, e_old, delta_n)
        else:
            all_nodes, all_links, edges_added = build_realistic_network(
                nodes,
                links,
                flow,
                [],
                delta_n,
                delta_e,
                templates,
                fallback,
                candidate_edges=candidates,
                node_offset_factor=NODE_OFFSET_FACTOR,
                max_extra_per_new=MAX_EXTRA_PER_NEW,
                max_degree=MAX_DEGREE,
                random_seed=random_seed,
            )

        all_links = apply_mean_attributes_to_added_edges(nodes, links, all_links)

    out_node, out_link = scenario_output_paths(city_id, scenario, experiment_group)
    os.makedirs(os.path.dirname(out_node), exist_ok=True)

    all_nodes.to_csv(out_node, header=False, index=False)
    all_links.to_csv(out_link, header=False, index=False)

    n_new = len(all_nodes)
    e_new = len(all_links)
    delta_e_actual = e_new - e_old
    density_new = 2 * e_new / (n_new * (n_new - 1)) if n_new > 1 else 0.0
    avg_degree_new = 2 * e_new / n_new if n_new > 0 else 0.0

    if revision_strategy == "proportional" and scenario == "modified_degree" and not ALLOW_PARTIAL_DEGREE:
        achieved = avg_degree_new / avg_degree_old - 1.0 if avg_degree_old > 0 else 0.0
        if achieved + TARGET_TOLERANCE < TARGET_METRIC_CHANGE:
            remove_revision_files(city_id, scenario, experiment_group)
            raise SkippedRevision(
                city_id,
                scenario,
                experiment_group,
                (
                    f"target +{TARGET_METRIC_CHANGE*100:.1f}% degree not reachable "
                    f"under max_degree={MAX_DEGREE}; achieved {achieved*100:.4f}%"
                ),
            )

    result = RevisionResult(
        city_id=city_id,
        scenario=scenario,
        experiment_group=experiment_group,
        neighbor_seed=neighbor_seed,
        n_old=n_old,
        e_old=e_old,
        n_new=n_new,
        e_new=e_new,
        delta_n=delta_n,
        delta_e=delta_e,
        delta_e_actual=delta_e_actual,
        density_old=density_old,
        density_new=density_new,
        avg_degree_old=avg_degree_old,
        avg_degree_new=avg_degree_new,
    )

    return RevisionOutput(
        result=result,
        original_node=node_path,
        original_link=link_path,
        revised_node=out_node,
        revised_link=out_link,
        scenario=scenario,
        city_id=city_id,
        experiment_group=experiment_group,
    )


def choose_revision_strategy() -> str:
    print("Select revision strategy:")
    print("  1. edge_count       - previous behavior: delta_n is 10% of congested edges")
    print("  2. congested_edges  - split top-X% congested edges, then add edges to satisfy controls")
    print("  3. proportional     - topology scenarios target a 20% network-level change")
    choice = input("Enter 1, 2 or 3 [default: 1]: ").strip()
    if choice == "2":
        return "congested_edges"
    if choice == "3":
        return "proportional"
    return "edge_count"


def choose_city_id() -> int | None:
    raw = input("Enter city id to run one city, or press Enter to run all: ").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid city id {raw!r}; running all cities.")
        return None


def main() -> None:
    global ALLOW_PARTIAL_DEGREE, OUTPUT_DIR, RANKING_OD
    parser = argparse.ArgumentParser(description="Generate revised network variants.")
    parser.add_argument(
        "--strategy",
        choices=["edge_count", "congested_edges", "proportional"],
        help="Revision strategy. If omitted, prompt interactively.",
    )
    parser.add_argument("--city-id", type=int, help="Run one city only.")
    parser.add_argument("--city-ids", help="Comma-separated city ids to run.")
    parser.add_argument("--all-cities", action="store_true", help="Run all available cities without prompting.")
    parser.add_argument(
        "--experiment-group",
        choices=[experiment_group for experiment_group, _ in NEIGHBOR_EXPERIMENTS],
        help="Run one neighbor-seed experiment group only.",
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated subset of scenarios (modified_density,modified_degree,modified_capacity,modified_speed_limit). Default: all four.",
    )
    parser.add_argument(
        "--allow-partial-degree",
        action="store_true",
        help="Keep modified_degree outputs even when the achieved average-degree increase is below TARGET_METRIC_CHANGE.",
    )
    parser.add_argument(
        "--change-ratio",
        type=float,
        default=TARGET_METRIC_CHANGE,
        help="Modification ratio for all four indicators. Use 0.1 or 10 for 10%%; 0.2 or 20 for 20%%.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for revised-network generation outputs. Defaults to auto_revise/output.",
    )
    parser.add_argument(
        "--ranking-od",
        type=int,
        default=RANKING_OD,
        choices=range(1, 11),
        metavar="{1..10}",
        help="Use this OD multiplier's original flow for V/C ranking. Default: 10.",
    )
    args = parser.parse_args()
    ALLOW_PARTIAL_DEGREE = args.allow_partial_degree
    RANKING_OD = args.ranking_od
    configure_change_ratio(args.change_ratio)
    if args.output_dir:
        OUTPUT_DIR = os.path.abspath(args.output_dir)

    city_id_modes = sum(
        1
        for selected in (args.city_id is not None, bool(args.city_ids), args.all_cities)
        if selected
    )
    if city_id_modes > 1:
        parser.error("--city-id, --city-ids, and --all-cities cannot be used together")

    city_id_list = None
    if args.city_ids:
        try:
            city_id_list = [int(value.strip()) for value in args.city_ids.split(",") if value.strip()]
        except ValueError as exc:
            parser.error(f"--city-ids must contain integers only: {exc}")

    revision_strategy = args.strategy or choose_revision_strategy()
    if city_id_list is not None:
        city_id_filter = None
    elif args.all_cities:
        city_id_filter = None
    elif args.city_id is not None:
        city_id_filter = args.city_id
    else:
        city_id_filter = choose_city_id()
    print(f"Revision strategy: {revision_strategy}")
    print(f"Change ratio: {TARGET_METRIC_CHANGE*100:.4g}%")
    print(f"Flow ranking OD: {RANKING_OD}")
    print("Flow fallback: disabled; requested OD flow is required")
    print(f"Output directory: {OUTPUT_DIR}")
    if city_id_list is not None:
        print(f"City selection: {len(city_id_list)} ids")
    elif city_id_filter is None:
        print("City selection: all")
    else:
        print(f"City selection: {city_id_filter}")
    if args.experiment_group is None:
        print("Experiment group selection: all")
    else:
        print(f"Experiment group selection: {args.experiment_group}")

    results: List[RevisionResult] = []
    outputs: List[RevisionOutput] = []
    skipped_rows: List[Dict[str, object]] = []
    reused_attribute_cases = 0

    tasks: List[Tuple[int, str, str, int]] = []
    if city_id_list is not None:
        city_ids = city_id_list
    else:
        city_ids = [city_id_filter] if city_id_filter is not None else range(1, 141)
    for city_id in city_ids:
        node_path = os.path.join(NETWORK_DIR, f"{city_id}-original_node.csv")
        link_path = os.path.join(NETWORK_DIR, f"{city_id}-original_link.csv")
        flow_path = find_flow_path(city_id)
        if not (os.path.exists(node_path) and os.path.exists(link_path) and os.path.exists(flow_path)):
            continue
        for experiment_group, neighbor_seed in NEIGHBOR_EXPERIMENTS:
            if args.experiment_group is not None and experiment_group != args.experiment_group:
                continue
            all_scenarios = [
                "modified_density",
                "modified_degree",
                "modified_capacity",
                "modified_speed_limit",
            ]
            if args.scenarios:
                requested = {s.strip() for s in args.scenarios.split(",") if s.strip()}
                scenarios = [s for s in all_scenarios if s in requested]
            else:
                scenarios = all_scenarios
            for scenario in scenarios:
                tasks.append((city_id, scenario, experiment_group, neighbor_seed))

    for city_id, scenario, experiment_group, neighbor_seed in tqdm(tasks, desc="Revising", unit="case"):
        output = reuse_reference_attribute_revision(
            city_id,
            scenario,
            experiment_group,
            neighbor_seed,
        )
        if output is not None:
            reused_attribute_cases += 1
        else:
            try:
                output = revise_city(
                    city_id,
                    scenario,
                    revision_strategy,
                    experiment_group=experiment_group,
                    neighbor_seed=neighbor_seed,
                )
            except SkippedRevision as exc:
                skipped_rows.append(
                    {
                        "city_id": exc.city_id,
                        "scenario": exc.scenario,
                        "experiment_group": exc.experiment_group,
                        "neighbor_seed": neighbor_seed,
                        "reason": exc.reason,
                    }
                )
                continue
        results.append(output.result)
        outputs.append(output)

    if reused_attribute_cases:
        print(
            f"Reused {REFERENCE_EXPERIMENT_GROUP} files for "
            f"{reused_attribute_cases} capacity/speed-limit cases."
        )

    if not results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        skipped_name = "degree_skipped_cities.csv"
        if args.experiment_group is not None:
            skipped_name = f"degree_skipped_cities_{args.experiment_group}.csv"
        skipped_path = os.path.join(OUTPUT_DIR, skipped_name)
        pd.DataFrame(
            skipped_rows,
            columns=["city_id", "scenario", "experiment_group", "neighbor_seed", "reason"],
        ).to_csv(skipped_path, index=False)
        if skipped_rows:
            print(f"Skipped {len(skipped_rows)} degree cases; details: {skipped_path}")
        checked_flow = find_flow_path(1)
        print(
            "No revision tasks found. Check NETWORK_DIR/FLOW_DIR and file naming. "
            f"Example expected files: {os.path.join(NETWORK_DIR, '1-original_node.csv')}, "
            f"{os.path.join(NETWORK_DIR, '1-original_link.csv')}, {checked_flow}"
        )
        return

    summary = pd.DataFrame([r.__dict__ for r in results])
    tol = 1e-6
    summary["density_match"] = (summary["density_new"] - summary["density_old"]).abs() <= tol
    summary["avg_degree_match"] = (summary["avg_degree_new"] - summary["avg_degree_old"]).abs() <= tol
    summary["delta_e_match"] = summary["delta_e_actual"] == summary["delta_e"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_name = "revision_summary.csv"
    if args.experiment_group is not None:
        summary_name = f"revision_summary_{args.experiment_group}.csv"
    summary_path = os.path.join(OUTPUT_DIR, summary_name)
    summary.to_csv(summary_path, index=False)

    skipped_name = "degree_skipped_cities.csv"
    if args.experiment_group is not None:
        skipped_name = f"degree_skipped_cities_{args.experiment_group}.csv"
    skipped_path = os.path.join(OUTPUT_DIR, skipped_name)
    skipped_df = pd.DataFrame(
        skipped_rows,
        columns=["city_id", "scenario", "experiment_group", "neighbor_seed", "reason"],
    )
    skipped_df.to_csv(skipped_path, index=False)
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} degree cases; details: {skipped_path}")


if __name__ == "__main__":
    main()
