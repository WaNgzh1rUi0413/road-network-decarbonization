import unittest

import numpy as np
import pandas as pd

from src.emissions.compute_emissions import calculate_emission
from src.modeling.train_models import FEATURES, OD_LEVELS
from src.od_selection.k_center_solver import KCenterSolver
from src.paths import NETWORK_DIR, PROCESSED_DATA
from src.preprocessing.topology_metrics import calculate_city_metrics


class CoreExperimentTests(unittest.TestCase):
    def test_processed_data_schema(self):
        data = pd.read_excel(PROCESSED_DATA, sheet_name="Sheet3")
        self.assertEqual(len(data), 140)
        self.assertTrue(set(FEATURES).issubset(data.columns))
        self.assertTrue(all(f"Normalized carbon emissions_{od}" in data for od in OD_LEVELS))
        self.assertTrue(data[FEATURES].notna().all().all())

    def test_all_versioned_networks_are_present(self):
        self.assertEqual(len(list(NETWORK_DIR.glob("*-original_node.csv"))), 140)
        self.assertEqual(len(list(NETWORK_DIR.glob("*-original_link.csv"))), 140)

    def test_structural_descriptors_match_versioned_table(self):
        expected = pd.read_excel(PROCESSED_DATA, sheet_name="Sheet3").set_index("NO").loc[1]
        calculated = calculate_city_metrics(1)
        for name, value in calculated.items():
            if name != "NO":
                self.assertAlmostEqual(value, expected[name], places=10, msg=name)

    def test_emission_formula_reference_value(self):
        expected = (0.0000236 * 36**2 - 0.00430 * 36 + 0.317) * 2 * 100
        self.assertEqual(calculate_emission(36, 2, 100), expected)

    def test_k_center_returns_requested_number_of_centers(self):
        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [4, 4]], dtype=float)
        centers, radius, indices = KCenterSolver(k=2, random_seed=42).solve_greedy(points, initial_center_idx=0)
        self.assertEqual(centers.shape, (2, 2))
        self.assertEqual(len(indices), 2)
        self.assertGreaterEqual(radius, 0)


if __name__ == "__main__":
    unittest.main()
