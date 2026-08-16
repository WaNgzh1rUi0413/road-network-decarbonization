import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _euclidean_length(coord: pd.DataFrame, u: int, v: int) -> float:
    x1, y1 = coord.loc[u]
    x2, y2 = coord.loc[v]
    return float(math.hypot(x2 - x1, y2 - y1))


def _safe_coord(coord: pd.DataFrame, node_id: int) -> Tuple[float, float] | None:
    if node_id in coord.index:
        return float(coord.loc[node_id][0]), float(coord.loc[node_id][1])
    return None


def _free_flow_time(length: float, speed_limit: float, fallback: float) -> float:
    if speed_limit and speed_limit > 0:
        return float(length / speed_limit / 3600.0)
    return float(fallback)


def _realistic_extra_edge_radius(nodes: pd.DataFrame, links: pd.DataFrame, diag: float) -> float:
    coord = nodes.set_index("node_id")[["x", "y"]]
    lengths: List[float] = []
    for u, v in links[["from", "to"]].values:
        u = int(u)
        v = int(v)
        if u not in coord.index or v not in coord.index:
            continue
        length = _euclidean_length(coord, u, v)
        if length > 0:
            lengths.append(length)

    if not lengths:
        return max(1.0, diag * 0.05)

    length_values = np.asarray(lengths, dtype=float)
    median_length = float(np.median(length_values))
    q90_length = float(np.quantile(length_values, 0.90))
    local_cap = max(1.0, median_length * 3.0, q90_length * 1.5)
    city_cap = max(1.0, diag * 0.12)
    return min(local_cap, city_cap)


def _build_degree_map(links: pd.DataFrame) -> Dict[int, int]:
    degree_map: Dict[int, int] = {}
    seen_edges = set()
    for u, v in links[["from", "to"]].values:
        u = int(u)
        v = int(v)
        key = _edge_key(u, v)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        degree_map[u] = degree_map.get(u, 0) + 1
        degree_map[v] = degree_map.get(v, 0) + 1
    return degree_map


def _build_neighbor_map(links: pd.DataFrame) -> Dict[int, List[int]]:
    neighbor_map: Dict[int, List[int]] = {}
    for u, v in links[["from", "to"]].values:
        u = int(u)
        v = int(v)
        neighbor_map.setdefault(u, []).append(v)
        neighbor_map.setdefault(v, []).append(u)
    return neighbor_map


def _add_neighbor(neighbor_map: Dict[int, List[int]], u: int, v: int) -> None:
    neighbors = neighbor_map.setdefault(u, [])
    if v not in neighbors:
        neighbors.append(v)


def _remove_neighbor(neighbor_map: Dict[int, List[int]], u: int, v: int) -> None:
    if u not in neighbor_map:
        return
    neighbors = neighbor_map[u]
    try:
        neighbors.remove(v)
    except ValueError:
        return


def _target_order_for_point(
    x: float,
    y: float,
    coord: pd.DataFrame,
    preferred: Iterable[int],
) -> Tuple[List[int], Dict[int, float]]:
    all_nodes = coord.index.to_numpy()
    all_xy = coord.to_numpy()
    deltas = all_xy - np.array([x, y])
    dists = np.sqrt((deltas ** 2).sum(axis=1))

    dist_map: Dict[int, float] = {}
    for i, node in enumerate(all_nodes):
        dist_map[int(node)] = float(dists[i])

    preferred_set = {int(v) for v in preferred}
    pref_idx = [i for i, node in enumerate(all_nodes) if int(node) in preferred_set]
    pref_idx.sort(key=lambda i: dists[i])
    preferred_targets = [int(all_nodes[i]) for i in pref_idx]

    order = np.argsort(dists)
    ordered_targets = []
    for i in order:
        node_id = int(all_nodes[i])
        if node_id in preferred_set:
            continue
        ordered_targets.append(node_id)

    return preferred_targets + ordered_targets, dist_map


def _weighted_target_order(
    targets: Iterable[int],
    dist_map: Dict[int, float],
    radius: float,
    rng: np.random.Generator,
) -> List[int]:
    remaining = list(dict.fromkeys(int(target) for target in targets))
    ordered: List[int] = []
    scale = max(float(radius), 1e-9)
    while remaining:
        distances = np.array([dist_map.get(target, float("inf")) for target in remaining], dtype=float)
        weights = np.exp(-3.0 * distances / scale)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(remaining), dtype=float)
        probabilities = weights / weights.sum()
        selected_idx = int(rng.choice(len(remaining), p=probabilities))
        ordered.append(remaining.pop(selected_idx))
    return ordered


def _has_available_target(
    ordered_targets: List[int],
    dist_map: Dict[int, float],
    exclude: Iterable[int],
    degree_map: Dict[int, int],
    max_degree: int,
    max_radius: float,
) -> bool:
    if max_degree <= 0:
        return True
    exclude_set = {int(v) for v in exclude}
    for target in ordered_targets:
        if target in exclude_set:
            continue
        if dist_map.get(target, float("inf")) > max_radius:
            continue
        if degree_map.get(target, 0) >= max_degree:
            continue
        return True
    return False


def _add_edges_for_new_node(
    coord: pd.DataFrame,
    new_id: int,
    ordered_targets: List[int],
    dist_map: Dict[int, float],
    radii: List[float],
    edge_set: set,
    degree_map: Dict[int, int],
    neighbor_map: Dict[int, List[int]],
    templates: Dict[int, object],
    fallback: object,
    max_degree: int,
    max_to_add: int,
    rng: np.random.Generator,
) -> Tuple[List[List[float]], int]:
    if max_to_add <= 0:
        return [], 0

    def template_for(u: int, v: int) -> object:
        return templates.get(v, templates.get(u, fallback))

    added = 0
    new_rows: List[List[float]] = []
    used_targets = set()

    for radius in radii:
        eligible_targets = [
            target
            for target in ordered_targets
            if target != new_id
            and target not in used_targets
            and dist_map.get(target, float("inf")) <= radius
        ]
        for target in _weighted_target_order(eligible_targets, dist_map, radius, rng):
            if added >= max_to_add:
                return new_rows, added
            key = _edge_key(new_id, target)
            if key in edge_set:
                continue
            if max_degree > 0:
                if degree_map.get(new_id, 0) >= max_degree:
                    continue
                if degree_map.get(target, 0) >= max_degree:
                    continue

            tpl = template_for(new_id, target)
            length = _euclidean_length(coord, new_id, target)
            fft = _free_flow_time(length, float(tpl.speed_limit), float(tpl.free_flow_time))
            new_rows.append(
                [
                    new_id,
                    target,
                    length,
                    tpl.capacity,
                    fft,
                    tpl.speed_limit,
                    tpl.attr6,
                    tpl.attr7,
                ]
            )
            edge_set.add(key)
            degree_map[new_id] = degree_map.get(new_id, 0) + 1
            degree_map[target] = degree_map.get(target, 0) + 1
            _add_neighbor(neighbor_map, new_id, target)
            _add_neighbor(neighbor_map, target, new_id)
            used_targets.add(target)
            added += 1

    return new_rows, added


def _build_ratio_map(flow: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    flow = flow.copy()
    flow["ratio"] = flow["flow"] / flow["capacity"].replace(0, np.nan)
    flow["ratio"] = flow["ratio"].fillna(0.0)
    ratio_map: Dict[Tuple[int, int], float] = {}
    for a, b, r in flow[["from", "to", "ratio"]].values:
        ra = float(r)
        ratio_map[(int(a), int(b))] = ra
        ratio_map[(int(b), int(a))] = ra
    return ratio_map


def _build_edge_ratio_list(flow: pd.DataFrame) -> List[Tuple[int, int, float]]:
    flow = flow.copy()
    flow["ratio"] = flow["flow"] / flow["capacity"].replace(0, np.nan)
    flow["ratio"] = flow["ratio"].fillna(0.0)
    records: Dict[Tuple[int, int], List[float]] = {}
    for a, b, r in flow[["from", "to", "ratio"]].values:
        u = int(a)
        v = int(b)
        key = _edge_key(u, v)
        records.setdefault(key, []).append(float(r))
    edge_list = []
    for (u, v), vals in records.items():
        edge_list.append((u, v, float(sum(vals) / len(vals))))
    edge_list.sort(key=lambda x: x[2], reverse=True)
    return edge_list


def _pick_candidate_edges_by_nodes(
    links: pd.DataFrame,
    base_nodes: List[int],
    ratio_map: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int, float]]:
    candidates = []
    base_set = set(base_nodes)
    for u, v in links[["from", "to"]].values:
        u = int(u)
        v = int(v)
        if u in base_set or v in base_set:
            ratio = ratio_map.get((u, v), ratio_map.get((v, u), 0.0))
            candidates.append((u, v, float(ratio)))
    if not candidates:
        for u, v in links[["from", "to"]].values:
            u = int(u)
            v = int(v)
            ratio = ratio_map.get((u, v), ratio_map.get((v, u), 0.0))
            candidates.append((u, v, float(ratio)))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _split_edges_for_nodes(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    candidates: List[Tuple[int, int, float]],
    delta_n: int,
    max_split_per_edge: int = 0,
    node_offset_factor: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], Dict[int, Tuple[int, int]], int]:
    nodes = nodes.copy()
    links = links.copy()
    coord = nodes.set_index("node_id")[["x", "y"]]
    max_id = int(nodes["node_id"].max())

    new_nodes: List[List[float]] = []
    new_ids: List[int] = []
    edges_added = 0
    split_map: Dict[int, Tuple[int, int]] = {}

    if not candidates:
        return nodes, links, new_ids, split_map, edges_added

    split_edges: Dict[Tuple[int, int], int] = {}
    target_count = min(delta_n, len(candidates))
    for u, v, _ in candidates:
        if len(new_nodes) >= target_count:
            break
        base_key = _edge_key(u, v)
        if max_split_per_edge > 0 and split_edges.get(base_key, 0) >= max_split_per_edge:
            continue

        mask = ((links["from"] == u) & (links["to"] == v)) | ((links["from"] == v) & (links["to"] == u))
        if not mask.any():
            continue
        edge_row = links[mask].iloc[0]
        links = links.drop(edge_row.name)

        x1, y1 = coord.loc[u]
        x2, y2 = coord.loc[v]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len > 0:
            ox, oy = -dy / seg_len, dx / seg_len
            offset = node_offset_factor * seg_len
        else:
            ox, oy = 0.0, 0.0
            offset = 0.0

        new_id = max_id + 1 + len(new_nodes)
        new_x = mx + ox * offset
        new_y = my + oy * offset
        new_nodes.append([new_id, new_x, new_y])
        new_ids.append(new_id)
        split_map[new_id] = (u, v)
        coord.loc[new_id] = [new_x, new_y]

        row1 = edge_row.copy()
        row1["from"] = u
        row1["to"] = new_id
        row1_len = _euclidean_length(coord, u, new_id)
        row1["length"] = row1_len
        row1["free_flow_time"] = _free_flow_time(
            row1_len,
            float(row1["speed_limit"]),
            float(row1["free_flow_time"]),
        )

        row2 = edge_row.copy()
        row2["from"] = new_id
        row2["to"] = v
        row2_len = _euclidean_length(coord, new_id, v)
        row2["length"] = row2_len
        row2["free_flow_time"] = _free_flow_time(
            row2_len,
            float(row2["speed_limit"]),
            float(row2["free_flow_time"]),
        )

        links = pd.concat([links, pd.DataFrame([row1, row2])], ignore_index=True)
        edges_added += 1
        split_edges[base_key] = split_edges.get(base_key, 0) + 1

    if new_nodes:
        nodes = pd.concat(
            [nodes, pd.DataFrame(new_nodes, columns=["node_id", "x", "y"])],
            ignore_index=True,
        )
    return nodes, links, new_ids, split_map, edges_added


def _add_rectangle_on_edge(
    u: int,
    v: int,
    coord: pd.DataFrame,
    next_id: int,
    size: float,
    degree_map: Dict[int, int],
    max_degree: int,
    templates: Dict[int, object],
    fallback: object,
    edge_set: set,
    flip: int,
) -> Tuple[List[List[float]], List[List[float]], int]:
    if max_degree > 0:
        if degree_map.get(u, 0) >= max_degree or degree_map.get(v, 0) >= max_degree:
            return [], [], next_id

    uv = _safe_coord(coord, u)
    vv = _safe_coord(coord, v)
    if uv is None or vv is None:
        return [], [], next_id
    x1, y1 = uv
    x2, y2 = vv
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len == 0:
        return [], [], next_id

    ox, oy = -dy / seg_len, dx / seg_len
    sign = 1.0 if flip % 2 == 0 else -1.0
    px, py = ox * size * sign, oy * size * sign

    n1 = next_id
    n2 = next_id + 1
    next_id += 2

    n1_x, n1_y = x1 + px, y1 + py
    n2_x, n2_y = x2 + px, y2 + py

    new_nodes = [[n1, n1_x, n1_y], [n2, n2_x, n2_y]]
    local_coord = {
        u: (x1, y1),
        v: (x2, y2),
        n1: (n1_x, n1_y),
        n2: (n2_x, n2_y),
    }

    def template_for(a: int, b: int) -> object:
        return templates.get(a, templates.get(b, fallback))

    new_edges = []
    for a, b in [(u, n1), (n1, n2), (n2, v)]:
        key = _edge_key(a, b)
        if key in edge_set:
            continue
        if max_degree > 0:
            if degree_map.get(a, 0) >= max_degree or degree_map.get(b, 0) >= max_degree:
                continue
        tpl = template_for(a, b)
        if a in local_coord:
            ax, ay = local_coord[a]
        else:
            av = _safe_coord(coord, a)
            if av is None:
                continue
            ax, ay = av
        if b in local_coord:
            bx, by = local_coord[b]
        else:
            bv = _safe_coord(coord, b)
            if bv is None:
                continue
            bx, by = bv
        length = float(math.hypot(ax - bx, ay - by))
        fft = _free_flow_time(length, float(tpl.speed_limit), float(tpl.free_flow_time))
        new_edges.append([a, b, length, tpl.capacity, fft, tpl.speed_limit, tpl.attr6, tpl.attr7])
        edge_set.add(key)
        degree_map[a] = degree_map.get(a, 0) + 1
        degree_map[b] = degree_map.get(b, 0) + 1

    return new_nodes, new_edges, next_id


def _add_local_edges(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    new_ids: List[int],
    extra_edges: int,
    templates: Dict[int, object],
    fallback: object,
    radius: float,
    max_per_new: int,
    split_map: Dict[int, Tuple[int, int]],
    neighbor_map: Dict[int, List[int]],
    per_new_added: Dict[int, int],
    degree_map: Dict[int, int],
    max_degree: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, int]:
    if extra_edges <= 0 or not new_ids:
        return links, 0

    coord = nodes.set_index("node_id")[["x", "y"]]
    existing = set(_edge_key(int(a), int(b)) for a, b in links[["from", "to"]].values)

    def template_for(u: int, v: int) -> object:
        return templates.get(v, templates.get(u, fallback))

    added = 0
    new_rows = []

    all_nodes = coord.index.to_numpy()
    all_xy = coord.to_numpy()

    for new_id in new_ids:
        if added >= extra_edges:
            break
        if new_id not in coord.index:
            continue
        nx, ny = coord.loc[new_id]
        deltas = all_xy - np.array([nx, ny])
        dists = np.sqrt((deltas ** 2).sum(axis=1))

        preferred_targets: List[int] = []
        if new_id in split_map:
            u, v = split_map[new_id]
            preferred = set(neighbor_map.get(u, [])) | set(neighbor_map.get(v, []))
            preferred.discard(u)
            preferred.discard(v)
            preferred.discard(new_id)
            if preferred:
                pref_idx = [np.where(all_nodes == t)[0][0] for t in preferred if t in set(all_nodes)]
                dist_map = {int(all_nodes[i]): float(dists[i]) for i in range(len(all_nodes))}
                preferred_targets = _weighted_target_order(
                    [int(all_nodes[i]) for i in pref_idx],
                    dist_map,
                    radius,
                    rng,
                )

        dist_map = {int(all_nodes[i]): float(dists[i]) for i in range(len(all_nodes))}
        order = [
            int(np.where(all_nodes == target)[0][0])
            for target in _weighted_target_order(
                [int(all_nodes[i]) for i in np.where(dists <= radius)[0]],
                dist_map,
                radius,
                rng,
            )
        ]

        count = per_new_added.get(new_id, 0)
        if max_per_new > 0 and count >= max_per_new:
            continue
        for target in preferred_targets:
            if added >= extra_edges or (max_per_new > 0 and count >= max_per_new):
                break
            idx = int(np.where(all_nodes == target)[0][0])
            if dists[idx] > radius:
                continue
            key = _edge_key(new_id, target)
            if key in existing:
                continue
            if max_degree > 0:
                if degree_map.get(new_id, 0) >= max_degree or degree_map.get(target, 0) >= max_degree:
                    continue
            tpl = template_for(new_id, target)
            length = _euclidean_length(coord, new_id, target)
            fft = _free_flow_time(length, float(tpl.speed_limit), float(tpl.free_flow_time))
            new_rows.append(
                [
                    new_id,
                    target,
                    length,
                    tpl.capacity,
                    fft,
                    tpl.speed_limit,
                    tpl.attr6,
                    tpl.attr7,
                ]
            )
            existing.add(key)
            added += 1
            count += 1
            per_new_added[new_id] = count
            degree_map[new_id] = degree_map.get(new_id, 0) + 1
            degree_map[target] = degree_map.get(target, 0) + 1

        if added >= extra_edges or (max_per_new > 0 and count >= max_per_new):
            continue

        for idx in order:
            if added >= extra_edges or (max_per_new > 0 and count >= max_per_new):
                break
            target = int(all_nodes[idx])
            if target == new_id:
                continue
            if dists[idx] > radius:
                break
            key = _edge_key(new_id, target)
            if key in existing:
                continue
            if max_degree > 0:
                if degree_map.get(new_id, 0) >= max_degree or degree_map.get(target, 0) >= max_degree:
                    continue
            tpl = template_for(new_id, target)
            length = _euclidean_length(coord, new_id, target)
            fft = _free_flow_time(length, float(tpl.speed_limit), float(tpl.free_flow_time))
            new_rows.append(
                [
                    new_id,
                    target,
                    length,
                    tpl.capacity,
                    fft,
                    tpl.speed_limit,
                    tpl.attr6,
                    tpl.attr7,
                ]
            )
            existing.add(key)
            added += 1
            count += 1
            per_new_added[new_id] = count
            degree_map[new_id] = degree_map.get(new_id, 0) + 1
            degree_map[target] = degree_map.get(target, 0) + 1

        if added >= extra_edges:
            break

    if new_rows:
        links = pd.concat(
            [links, pd.DataFrame(new_rows, columns=links.columns)],
            ignore_index=True,
        )
    return links, added


def _add_global_edges(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    new_ids: List[int],
    extra_edges: int,
    templates: Dict[int, object],
    fallback: object,
    max_per_new: int,
    split_map: Dict[int, Tuple[int, int]],
    neighbor_map: Dict[int, List[int]],
    per_new_added: Dict[int, int],
    degree_map: Dict[int, int],
    max_degree: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, int]:
    if extra_edges <= 0 or not new_ids:
        return links, 0

    coord = nodes.set_index("node_id")[["x", "y"]]
    existing = set(_edge_key(int(a), int(b)) for a, b in links[["from", "to"]].values)

    def template_for(u: int, v: int) -> object:
        return templates.get(v, templates.get(u, fallback))

    added = 0
    new_rows = []

    all_nodes = coord.index.to_numpy()
    all_xy = coord.to_numpy()

    for new_id in new_ids:
        if added >= extra_edges:
            break
        if new_id not in coord.index:
            continue
        nx, ny = coord.loc[new_id]
        deltas = all_xy - np.array([nx, ny])
        dists = np.sqrt((deltas ** 2).sum(axis=1))
        preferred_targets: List[int] = []
        if new_id in split_map:
            u, v = split_map[new_id]
            preferred = set(neighbor_map.get(u, [])) | set(neighbor_map.get(v, []))
            preferred.discard(u)
            preferred.discard(v)
            preferred.discard(new_id)
            if preferred:
                pref_idx = [np.where(all_nodes == t)[0][0] for t in preferred if t in set(all_nodes)]
                dist_map = {int(all_nodes[i]): float(dists[i]) for i in range(len(all_nodes))}
                radius = max(float(dists.max()), 1.0)
                preferred_targets = _weighted_target_order(
                    [int(all_nodes[i]) for i in pref_idx],
                    dist_map,
                    radius,
                    rng,
                )

        dist_map = {int(all_nodes[i]): float(dists[i]) for i in range(len(all_nodes))}
        radius = max(float(dists.max()), 1.0)
        order = [
            int(np.where(all_nodes == target)[0][0])
            for target in _weighted_target_order(all_nodes, dist_map, radius, rng)
        ]

        count = per_new_added.get(new_id, 0)
        if max_per_new > 0 and count >= max_per_new:
            continue
        for target in preferred_targets:
            if added >= extra_edges or (max_per_new > 0 and count >= max_per_new):
                break
            idx = int(np.where(all_nodes == target)[0][0])
            key = _edge_key(new_id, target)
            if key in existing:
                continue
            if max_degree > 0:
                if degree_map.get(new_id, 0) >= max_degree or degree_map.get(target, 0) >= max_degree:
                    continue
            tpl = template_for(new_id, target)
            length = _euclidean_length(coord, new_id, target)
            fft = _free_flow_time(length, float(tpl.speed_limit), float(tpl.free_flow_time))
            new_rows.append(
                [
                    new_id,
                    target,
                    length,
                    tpl.capacity,
                    fft,
                    tpl.speed_limit,
                    tpl.attr6,
                    tpl.attr7,
                ]
            )
            existing.add(key)
            added += 1
            count += 1
            per_new_added[new_id] = count
            degree_map[new_id] = degree_map.get(new_id, 0) + 1
            degree_map[target] = degree_map.get(target, 0) + 1

        if added >= extra_edges or (max_per_new > 0 and count >= max_per_new):
            continue

        for idx in order:
            if added >= extra_edges or (max_per_new > 0 and count >= max_per_new):
                break
            target = int(all_nodes[idx])
            if target == new_id:
                continue
            key = _edge_key(new_id, target)
            if key in existing:
                continue
            if max_degree > 0:
                if degree_map.get(new_id, 0) >= max_degree or degree_map.get(target, 0) >= max_degree:
                    continue
            tpl = template_for(new_id, target)
            length = _euclidean_length(coord, new_id, target)
            fft = _free_flow_time(length, float(tpl.speed_limit), float(tpl.free_flow_time))
            new_rows.append(
                [
                    new_id,
                    target,
                    length,
                    tpl.capacity,
                    fft,
                    tpl.speed_limit,
                    tpl.attr6,
                    tpl.attr7,
                ]
            )
            existing.add(key)
            added += 1
            count += 1
            per_new_added[new_id] = count
            degree_map[new_id] = degree_map.get(new_id, 0) + 1
            degree_map[target] = degree_map.get(target, 0) + 1

        if added >= extra_edges:
            break

    if new_rows:
        links = pd.concat(
            [links, pd.DataFrame(new_rows, columns=links.columns)],
            ignore_index=True,
        )
    return links, added


def build_realistic_network(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    flow: pd.DataFrame,
    base_nodes: List[int],
    delta_n: int,
    delta_e: int,
    templates: Dict[int, object],
    fallback: object,
    candidate_edges: Optional[List[Tuple[int, int, float]]] = None,
    radius_factor: float = 0.05,
    node_offset_factor: float = 0.10,
    max_extra_per_new: int = 3,
    max_degree: int = 4,
    random_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(random_seed)
    if candidate_edges:
        candidates = candidate_edges
    else:
        ratio_map = _build_ratio_map(flow)
        candidates = _pick_candidate_edges_by_nodes(links, base_nodes, ratio_map)

    nodes_updated = nodes.copy()
    links_updated = links.copy()
    coord = nodes_updated.set_index("node_id")[["x", "y"]]
    max_id = int(nodes_updated["node_id"].max()) if not nodes_updated.empty else 0

    edge_set = set(_edge_key(int(a), int(b)) for a, b in links_updated[["from", "to"]].values)
    degree_map = _build_degree_map(links_updated)
    neighbor_map = _build_neighbor_map(links_updated)

    new_nodes: List[List[float]] = []
    new_ids: List[int] = []
    split_map: Dict[int, Tuple[int, int]] = {}
    edges_added = 0

    if not nodes_updated.empty:
        xy = nodes_updated[["x", "y"]].to_numpy()
        min_xy = xy.min(axis=0)
        max_xy = xy.max(axis=0)
        diag = float(np.linalg.norm(max_xy - min_xy))
    else:
        diag = 1.0

    radii = [max(1.0, diag * radius_factor * scale) for scale in [1.0, 2.0, 4.0, 8.0]]

    for u, v, _ in candidates:
        if len(new_nodes) >= delta_n:
            break

        mask = ((links_updated["from"] == u) & (links_updated["to"] == v)) | (
            (links_updated["from"] == v) & (links_updated["to"] == u)
        )
        if not mask.any():
            continue

        extra_remaining = max(0, int(delta_e) - edges_added - 1)
        per_node_limit = max_extra_per_new
        if max_degree > 0:
            per_node_limit = min(per_node_limit, max_degree - 2)
        if per_node_limit < 0:
            per_node_limit = 0

        remaining_nodes = max(0, delta_n - (len(new_nodes) + 1))
        if edges_added + 1 + remaining_nodes > delta_e:
            continue
        max_extra_for_this = max(0, int(delta_e) - (edges_added + 1) - remaining_nodes)
        allowed_extra = min(per_node_limit, extra_remaining, max_extra_for_this)

        if extra_remaining > 0 and per_node_limit == 0 and allowed_extra > 0:
            continue

        x1, y1 = coord.loc[u]
        x2, y2 = coord.loc[v]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len > 0:
            ox, oy = -dy / seg_len, dx / seg_len
            offset = node_offset_factor * seg_len
        else:
            ox, oy = 0.0, 0.0
            offset = 0.0

        new_id = max_id + 1 + len(new_nodes)
        new_x = mx + ox * offset
        new_y = my + oy * offset

        preferred = set(neighbor_map.get(u, [])) | set(neighbor_map.get(v, []))
        preferred.discard(u)
        preferred.discard(v)
        preferred_targets, dist_map = _target_order_for_point(new_x, new_y, coord, preferred)
        if allowed_extra > 0:
            if not _has_available_target(
                preferred_targets,
                dist_map,
                [u, v, new_id],
                degree_map,
                max_degree,
                radii[-1],
            ):
                continue

        edge_row = links_updated[mask].iloc[0]
        links_updated = links_updated.drop(edge_row.name)
        old_key = _edge_key(u, v)
        if old_key in edge_set:
            edge_set.remove(old_key)
            degree_map[u] = degree_map.get(u, 0) - 1
            degree_map[v] = degree_map.get(v, 0) - 1
            _remove_neighbor(neighbor_map, u, v)
            _remove_neighbor(neighbor_map, v, u)

        coord.loc[new_id] = [new_x, new_y]
        new_nodes.append([new_id, new_x, new_y])
        new_ids.append(new_id)
        split_map[new_id] = (u, v)

        row1 = edge_row.copy()
        row1["from"] = u
        row1["to"] = new_id
        row1_len = _euclidean_length(coord, u, new_id)
        row1["length"] = row1_len
        row1["free_flow_time"] = _free_flow_time(
            row1_len,
            float(row1["speed_limit"]),
            float(row1["free_flow_time"]),
        )

        row2 = edge_row.copy()
        row2["from"] = new_id
        row2["to"] = v
        row2_len = _euclidean_length(coord, new_id, v)
        row2["length"] = row2_len
        row2["free_flow_time"] = _free_flow_time(
            row2_len,
            float(row2["speed_limit"]),
            float(row2["free_flow_time"]),
        )

        new_edge_rows: List[List[float]] = []
        new_edge_rows.append(
            [
                int(row1["from"]),
                int(row1["to"]),
                float(row1["length"]),
                float(row1["capacity"]),
                float(row1["free_flow_time"]),
                float(row1["speed_limit"]),
                float(row1["attr6"]),
                float(row1["attr7"]),
            ]
        )
        new_edge_rows.append(
            [
                int(row2["from"]),
                int(row2["to"]),
                float(row2["length"]),
                float(row2["capacity"]),
                float(row2["free_flow_time"]),
                float(row2["speed_limit"]),
                float(row2["attr6"]),
                float(row2["attr7"]),
            ]
        )

        edge_set.add(_edge_key(u, new_id))
        edge_set.add(_edge_key(new_id, v))
        degree_map[u] = degree_map.get(u, 0) + 1
        degree_map[v] = degree_map.get(v, 0) + 1
        degree_map[new_id] = degree_map.get(new_id, 0) + 2
        _add_neighbor(neighbor_map, u, new_id)
        _add_neighbor(neighbor_map, new_id, u)
        _add_neighbor(neighbor_map, v, new_id)
        _add_neighbor(neighbor_map, new_id, v)

        edges_added += 1

        if allowed_extra > 0:
            ordered_targets, dist_map = _target_order_for_point(new_x, new_y, coord, preferred)
            added_rows, added = _add_edges_for_new_node(
                coord,
                new_id,
                ordered_targets,
                dist_map,
                radii,
                edge_set,
                degree_map,
                neighbor_map,
                templates,
                fallback,
                max_degree,
                allowed_extra,
                rng,
            )
            if added_rows:
                new_edge_rows.extend(added_rows)
            edges_added += added

        if new_edge_rows:
            links_updated = pd.concat(
                [links_updated, pd.DataFrame(new_edge_rows, columns=links_updated.columns)],
                ignore_index=True,
            )

    remaining_edges = max(0, int(delta_e) - edges_added)
    extra_edge_rows: List[List[float]] = []
    if remaining_edges > 0:
        for u, v, _ in candidates:
            if remaining_edges <= 0:
                break
            for node_id in (u, v):
                if remaining_edges <= 0:
                    break
                if node_id not in coord.index:
                    continue
                if max_degree > 0 and degree_map.get(node_id, 0) >= max_degree:
                    continue
                ordered_targets, dist_map = _target_order_for_point(
                    float(coord.loc[node_id][0]),
                    float(coord.loc[node_id][1]),
                    coord,
                    [],
                )
                allowed = remaining_edges
                if max_degree > 0:
                    allowed = min(allowed, max_degree - degree_map.get(node_id, 0))
                if allowed <= 0:
                    continue
                added_rows, added = _add_edges_for_new_node(
                    coord,
                    int(node_id),
                    ordered_targets,
                    dist_map,
                    radii,
                    edge_set,
                    degree_map,
                    neighbor_map,
                    templates,
                    fallback,
                    max_degree,
                    allowed,
                    rng,
                )
                if added_rows:
                    extra_edge_rows.extend(added_rows)
                edges_added += added
                remaining_edges -= added

    if extra_edge_rows:
        links_updated = pd.concat(
            [links_updated, pd.DataFrame(extra_edge_rows, columns=links_updated.columns)],
            ignore_index=True,
        )

    if new_nodes:
        nodes_updated = pd.concat(
            [nodes_updated, pd.DataFrame(new_nodes, columns=["node_id", "x", "y"])],
            ignore_index=True,
        )

    return nodes_updated, links_updated, edges_added


def build_congested_edge_network(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    candidates: List[Tuple[int, int, float]],
    split_count: int,
    delta_e: int,
    templates: Dict[int, object],
    fallback: object,
    radius_factor: float = 0.05,
    node_offset_factor: float = 0.10,
    max_extra_per_new: int = 3,
    max_degree: int = 4,
    random_seed: int = 0,
    relaxed_max_degree: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    rng = np.random.default_rng(random_seed)
    nodes_updated = nodes.copy()
    links_updated = links.copy()
    coord = nodes_updated.set_index("node_id")[["x", "y"]]
    max_id = int(nodes_updated["node_id"].max()) if not nodes_updated.empty else 0

    edge_set = set(_edge_key(int(a), int(b)) for a, b in links_updated[["from", "to"]].values)
    degree_map = _build_degree_map(links_updated)
    neighbor_map = _build_neighbor_map(links_updated)

    new_nodes: List[List[float]] = []
    new_ids: List[int] = []
    split_map: Dict[int, Tuple[int, int]] = {}
    edges_added = 0

    if not nodes_updated.empty:
        xy = nodes_updated[["x", "y"]].to_numpy()
        diag = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    else:
        diag = 1.0
    extra_edge_radius = _realistic_extra_edge_radius(nodes_updated, links_updated, diag)
    split_radius = max(1.0, diag * radius_factor)
    radii = [
        max(1.0, min(extra_edge_radius, split_radius * scale))
        for scale in [1.0, 1.5, 2.0, 3.0]
    ]

    if max_degree > 0 and max_degree < 2:
        return nodes_updated, links_updated, 0, 0

    for u, v, _ in candidates:
        if len(new_nodes) >= split_count:
            break

        mask = ((links_updated["from"] == u) & (links_updated["to"] == v)) | (
            (links_updated["from"] == v) & (links_updated["to"] == u)
        )
        if not mask.any():
            continue

        edge_row = links_updated[mask].iloc[0]
        links_updated = links_updated.drop(edge_row.name)
        old_key = _edge_key(u, v)
        if old_key in edge_set:
            edge_set.remove(old_key)
            degree_map[u] = degree_map.get(u, 0) - 1
            degree_map[v] = degree_map.get(v, 0) - 1
            _remove_neighbor(neighbor_map, u, v)
            _remove_neighbor(neighbor_map, v, u)

        x1, y1 = coord.loc[u]
        x2, y2 = coord.loc[v]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len > 0:
            ox, oy = -dy / seg_len, dx / seg_len
            offset = node_offset_factor * seg_len
        else:
            ox, oy = 0.0, 0.0
            offset = 0.0

        new_id = max_id + 1 + len(new_nodes)
        new_x = mx + ox * offset
        new_y = my + oy * offset
        coord.loc[new_id] = [new_x, new_y]
        new_nodes.append([new_id, new_x, new_y])
        new_ids.append(new_id)
        split_map[new_id] = (u, v)

        row1 = edge_row.copy()
        row1["from"] = u
        row1["to"] = new_id
        row1_len = _euclidean_length(coord, u, new_id)
        row1["length"] = row1_len
        row1["free_flow_time"] = _free_flow_time(
            row1_len,
            float(row1["speed_limit"]),
            float(row1["free_flow_time"]),
        )

        row2 = edge_row.copy()
        row2["from"] = new_id
        row2["to"] = v
        row2_len = _euclidean_length(coord, new_id, v)
        row2["length"] = row2_len
        row2["free_flow_time"] = _free_flow_time(
            row2_len,
            float(row2["speed_limit"]),
            float(row2["free_flow_time"]),
        )

        links_updated = pd.concat(
            [
                links_updated,
                pd.DataFrame(
                    [
                        [
                            int(row1["from"]),
                            int(row1["to"]),
                            float(row1["length"]),
                            float(row1["capacity"]),
                            float(row1["free_flow_time"]),
                            float(row1["speed_limit"]),
                            float(row1["attr6"]),
                            float(row1["attr7"]),
                        ],
                        [
                            int(row2["from"]),
                            int(row2["to"]),
                            float(row2["length"]),
                            float(row2["capacity"]),
                            float(row2["free_flow_time"]),
                            float(row2["speed_limit"]),
                            float(row2["attr6"]),
                            float(row2["attr7"]),
                        ],
                    ],
                    columns=links_updated.columns,
                ),
            ],
            ignore_index=True,
        )

        edge_set.add(_edge_key(u, new_id))
        edge_set.add(_edge_key(new_id, v))
        degree_map[u] = degree_map.get(u, 0) + 1
        degree_map[v] = degree_map.get(v, 0) + 1
        degree_map[new_id] = 2
        _add_neighbor(neighbor_map, u, new_id)
        _add_neighbor(neighbor_map, new_id, u)
        _add_neighbor(neighbor_map, v, new_id)
        _add_neighbor(neighbor_map, new_id, v)
        edges_added += 1

    if new_nodes:
        nodes_updated = pd.concat(
            [nodes_updated, pd.DataFrame(new_nodes, columns=["node_id", "x", "y"])],
            ignore_index=True,
        )

    remaining_edges = max(0, int(delta_e) - edges_added)
    per_new_added = {new_id: 0 for new_id in new_ids}
    while remaining_edges > 0:
        before = remaining_edges
        links_updated, added = _add_local_edges(
            nodes_updated,
            links_updated,
            new_ids,
            remaining_edges,
            templates,
            fallback,
            radii[-1],
            max_extra_per_new,
            split_map,
            neighbor_map,
            per_new_added,
            degree_map,
            max_degree,
            rng,
        )
        edges_added += added
        remaining_edges -= added
        if remaining_edges <= 0:
            break

        if remaining_edges == before:
            break

    # Optional soft-relaxation pass: only if still short, allow target node
    # degrees up to relaxed_max_degree to fill the remaining gap.
    if remaining_edges > 0 and relaxed_max_degree > max_degree:
        while remaining_edges > 0:
            before = remaining_edges
            links_updated, added = _add_local_edges(
                nodes_updated,
                links_updated,
                new_ids,
                remaining_edges,
                templates,
                fallback,
                radii[-1],
                max_extra_per_new,
                split_map,
                neighbor_map,
                per_new_added,
                degree_map,
                relaxed_max_degree,
                rng,
            )
            edges_added += added
            remaining_edges -= added
            if remaining_edges <= 0:
                break
            if remaining_edges == before:
                break

    return nodes_updated, links_updated, edges_added, len(new_ids)
