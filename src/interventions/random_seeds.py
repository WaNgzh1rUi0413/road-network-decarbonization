from __future__ import annotations

from typing import Tuple


NODE_OFFSET_FACTOR = 0.10
NEIGHBOR_SEEDS: Tuple[int, ...] = tuple(range(101, 111))
NEIGHBOR_EXPERIMENTS: Tuple[Tuple[str, int], ...] = tuple(
    (f"neighbor_seed_{index:02d}", seed)
    for index, seed in enumerate(NEIGHBOR_SEEDS, start=1)
)


def derive_random_seed(base_seed: int, city_id: int, scenario: str) -> int:
    scenario_index = {
        "modified_density": 1,
        "modified_degree": 2,
        "modified_capacity": 3,
        "modified_speed_limit": 4,
    }[scenario]
    return base_seed + city_id * 10_000 + scenario_index * 1_000_000
