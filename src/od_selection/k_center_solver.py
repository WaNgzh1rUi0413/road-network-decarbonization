"""K-center optimization routines for topology-aware OD-center selection."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist
import os
import sys
import gc


# Exact methods require the commercial COPT optimizer.
try:
    import coptpy as copt
    COPT_AVAILABLE = True
except ImportError:
    COPT_AVAILABLE = False
    print("Warning: COPT solver not available. Install with: pip install coptpy")


class KCenterSolver:
    """Select spatially distributed centers from planar node coordinates."""

    def __init__(self, k: int = 50, random_seed: Optional[int] = None):
        """Initialize the solver with a center count and reproducible seed."""
        self.k = k
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def compute_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Return the pairwise Euclidean distance matrix."""
        return cdist(points, points, metric='euclidean')

    def solve_greedy(self, points: np.ndarray, initial_center_idx: Optional[int] = None) -> Tuple[np.ndarray, float, List[int]]:
        """Run Gonzalez's greedy farthest-first approximation."""
        n = len(points)
        if n <= self.k:
            # Every point can be selected when the sample is no larger than k.
            return points.copy(), 0.0, list(range(n))

        dist_matrix = self.compute_distance_matrix(points)

        if initial_center_idx is not None:
            center_indices = [initial_center_idx]
        else:
            center_indices = [np.random.randint(0, n)]

        for _ in range(self.k - 1):
            min_distances = np.min(dist_matrix[center_indices, :], axis=0)
            farthest_idx = np.argmax(min_distances)
            center_indices.append(int(farthest_idx))

        center_dist_matrix = dist_matrix[center_indices, :]
        min_distances = np.min(center_dist_matrix, axis=0)
        max_distance = np.max(min_distances)

        centers = points[center_indices]
        return centers, max_distance, center_indices

    def solve_multiple_runs(self, points: np.ndarray, n_runs: int = 10) -> List[Tuple[np.ndarray, float, List[int]]]:
        """Repeat the randomized greedy solver and rank solutions by radius."""
        results = []
        for i in range(n_runs):
            if self.random_seed is not None:
                np.random.seed(self.random_seed + i)
            result = self.solve_greedy(points)
            results.append(result)

        results.sort(key=lambda x: x[1])
        return results

    def solve_exhaustive(self, points: np.ndarray) -> List[Tuple[np.ndarray, float, List[int]]]:
        """Use every node as the first center and rank all greedy solutions."""
        n = len(points)
        results = []

        print(f"Testing all {n} nodes as the initial center...")
        for i in range(n):
            if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
                print(f"  Progress: {i + 1}/{n} ({100 * (i + 1) / n:.1f}%)")
            result = self.solve_greedy(points, initial_center_idx=i)
            results.append(result)

        results.sort(key=lambda x: x[1])
        return results

    def find_optimal_solutions(self, points: np.ndarray, n_runs: int = 10,
                              tolerance: float = 1e-6,
                              use_exhaustive: bool = False,
                              use_milp: bool = False,
                              use_binary_search: bool = False) -> List[Tuple[np.ndarray, float, List[int]]]:
        """Return unique best solutions within the requested tolerance."""
        if use_binary_search:
            if not COPT_AVAILABLE:
                raise ImportError("COPT solver not available. Cannot use binary search feasibility method.")
            result = self.solve_binary_search_feasibility(points, tolerance=tolerance)
            return [result]
        elif use_milp:
            if not COPT_AVAILABLE:
                raise ImportError("COPT solver not available. Cannot use MILP method.")
            result = self.solve_milp(points)
            return [result]
        elif use_exhaustive:
            results = self.solve_exhaustive(points)
        else:
            results = self.solve_multiple_runs(points, n_runs)

        if not results:
            return []

        best_distance = results[0][1]

        optimal_solutions = [
            result for result in results
            if abs(result[1] - best_distance) <= tolerance
        ]

        unique_solutions = []
        seen_indices_sets = set()

        for result in optimal_solutions:
            indices_set = tuple(sorted(result[2]))
            if indices_set not in seen_indices_sets:
                seen_indices_sets.add(indices_set)
                unique_solutions.append(result)

        return unique_solutions

    def solve_milp(self, points: np.ndarray) -> Tuple[np.ndarray, float, List[int]]:
        """Solve the K-center problem as an exact mixed-integer program."""
        if not COPT_AVAILABLE:
            raise ImportError("COPT solver not available. Please install coptpy.")

        n = len(points)
        if n <= self.k:
            return points.copy(), 0.0, list(range(n))

        dist_matrix = self.compute_distance_matrix(points)

        env = copt.Envr()
        model = env.createModel("K-Center Problem")

        y = {}
        for j in range(n):
            y[j] = model.addVar(vtype=copt.COPT.BINARY, name=f"y_{j}")

        x = {}
        for i in range(n):
            for j in range(n):
                x[i, j] = model.addVar(vtype=copt.COPT.BINARY, name=f"x_{i}_{j}")

        W = model.addVar(lb=0.0, vtype=copt.COPT.CONTINUOUS, name="W")

        model.setObjective(W, sense=copt.COPT.MINIMIZE)

        model.addConstr(sum(y[j] for j in range(n)) == self.k, name="select_k_centers")

        for i in range(n):
            model.addConstr(sum(x[i, j] for j in range(n)) == 1, name=f"assign_node_{i}")

        for i in range(n):
            for j in range(n):
                model.addConstr(x[i, j] <= y[j], name=f"logic_{i}_{j}")

        for i in range(n):
            model.addConstr(
                sum(dist_matrix[i, j] * x[i, j] for j in range(n)) <= W,
                name=f"radius_{i}"
            )

        print("Solving the K-center MILP with COPT...")
        print(f"  Variables: {n * (n + 1) + 1}")
        print(f"  Constraints: {1 + n + n*n + n}")

        model.solve()

        status = model.status
        if status == copt.COPT.OPTIMAL:
            print("  Solver status: optimal")
        elif status == copt.COPT.TIME_OUT:
            print("  Warning: solver time limit reached")
        elif status == copt.COPT.INFEASIBLE:
            raise ValueError("The K-center MILP is infeasible")
        else:
            print(f"  Warning: solver status = {status}")

        max_distance = W.x
        center_indices = [j for j in range(n) if y[j].x > 0.5]

        if len(center_indices) != self.k:
            print(f"  Warning: selected {len(center_indices)} centers instead of {self.k}")

        centers = points[center_indices]

        center_dist_matrix = dist_matrix[center_indices, :]
        min_distances = np.min(center_dist_matrix, axis=0)
        actual_max_distance = np.max(min_distances)

        print(f"  Model objective: {max_distance:.6f}")
        print(f"  Verified covering radius: {actual_max_distance:.6f}")

        return centers, max_distance, center_indices

    def check_feasibility(self, points: np.ndarray, radius: float, dist_matrix: Optional[np.ndarray] = None) -> Tuple[bool, Optional[List[int]]]:
        """Test whether k centers cover every point within a fixed radius."""
        if not COPT_AVAILABLE:
            raise ImportError("COPT solver not available. Please install coptpy.")

        n = len(points)
        if n <= self.k:
            return True, list(range(n))

        if dist_matrix is None:
            dist_matrix = self.compute_distance_matrix(points)

        env = copt.Envr()
        model = env.createModel("K-Center Feasibility Check")

        y = {}
        for j in range(n):
            y[j] = model.addVar(vtype=copt.COPT.BINARY, name=f"y_{j}")

        model.addConstr(sum(y[j] for j in range(n)) == self.k, name="select_k_centers")

        for i in range(n):
            candidates = [j for j in range(n) if dist_matrix[i, j] <= radius]
            if not candidates:
                return False, None
            model.addConstr(sum(y[j] for j in candidates) >= 1, name=f"cover_node_{i}")

        model.setObjective(0, sense=copt.COPT.MINIMIZE)
        model.solve()

        status = model.status
        if status == copt.COPT.OPTIMAL:
            center_indices = [j for j in range(n) if y[j].x > 0.5]
            return True, center_indices
        elif status == copt.COPT.INFEASIBLE:
            return False, None
        else:
            return False, None

    def solve_binary_search_feasibility(self, points: np.ndarray,
                                       tolerance: float = 1e-6,
                                       max_iterations: int = 100) -> Tuple[np.ndarray, float, List[int]]:
        """Solve exactly by binary-searching the feasible covering radius."""
        if not COPT_AVAILABLE:
            raise ImportError("COPT solver not available. Please install coptpy.")

        n = len(points)
        if n <= self.k:
            return points.copy(), 0.0, list(range(n))

        dist_matrix = self.compute_distance_matrix(points)

        all_distances = dist_matrix[dist_matrix > 0]
        if len(all_distances) == 0:
            return points.copy(), 0.0, list(range(n))

        _, greedy_max_distance, _ = self.solve_greedy(points)
        upper_bound = greedy_max_distance
        lower_bound = 0.0

        print(f"Binary-search range: [{lower_bound:.6f}, {upper_bound:.6f}]")
        print("Solving radius-feasibility models with COPT...")

        best_radius = upper_bound
        best_center_indices = None
        iteration = 0

        while (upper_bound - lower_bound) > tolerance and iteration < max_iterations:
            iteration += 1
            mid_radius = (lower_bound + upper_bound) / 2.0

            is_feasible, center_indices = self.check_feasibility(points, mid_radius, dist_matrix)

            if is_feasible:
                upper_bound = mid_radius
                best_radius = mid_radius
                best_center_indices = center_indices
                print(f"  Iteration {iteration}: radius {mid_radius:.6f} is feasible")
            else:
                lower_bound = mid_radius
                print(f"  Iteration {iteration}: radius {mid_radius:.6f} is infeasible")

        if best_center_indices is None:
            print("  Warning: exact search found no feasible solution; using greedy result")
            _, best_radius, best_center_indices = self.solve_greedy(points)

        centers = points[best_center_indices]

        center_dist_matrix = dist_matrix[best_center_indices, :]
        min_distances = np.min(center_dist_matrix, axis=0)
        actual_max_distance = np.max(min_distances)

        print(f"Binary search completed after {iteration} iterations")
        print(f"  Optimal radius: {best_radius:.6f}")
        print(f"  Verified covering radius: {actual_max_distance:.6f}")

        return centers, actual_max_distance, best_center_indices


def load_node_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load node IDs and planar coordinates from a headerless CSV file."""
    # Ignore any trailing empty columns in source node files.
    df = pd.read_csv(filepath, header=None, usecols=[0, 1, 2], names=['node_id', 'x', 'y'])

    if df[['x', 'y']].isna().any().any():
        nan_count = df[['x', 'y']].isna().sum().sum()
        print(f"Warning: dropping {nan_count} missing coordinate values from {filepath}")
        df = df.dropna(subset=['x', 'y'])
        if len(df) == 0:
            raise ValueError(f"Invalid coordinate data in {filepath}")

    points = df[['x', 'y']].values.astype(float)
    node_ids = df['node_id'].values

    # Validate again after conversion to a numeric array.
    if np.isnan(points).any():
        nan_rows = np.where(np.isnan(points).any(axis=1))[0]
        print(f"Warning: dropping invalid coordinate rows {nan_rows[:10]} from {filepath}")
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]
        node_ids = node_ids[valid_mask]
        if len(points) == 0:
            raise ValueError(f"Invalid coordinate data in {filepath}")

    return points, node_ids


def save_results(centers: np.ndarray, center_indices: List[int], node_ids: np.ndarray,
                output_path: str, max_distance: float):
    """Save selected node IDs, coordinates, and the covering radius."""
    center_node_ids = node_ids[center_indices]

    result_df = pd.DataFrame({
        'node_id': center_node_ids,
        'index': center_indices,
        'x': centers[:, 0],
        'y': centers[:, 1],
        'max_min_distance': max_distance
    })

    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved K-center results to: {output_path}")


