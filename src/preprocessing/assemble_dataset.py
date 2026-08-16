"""Assemble network descriptors and emission outputs into the modeling table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.modeling.train_models import OD_LEVELS
from src.paths import PROCESSED_DATA, REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=PROCESSED_DATA)
    parser.add_argument("--descriptors", type=Path, default=REPO_ROOT / "outputs" / "network_descriptors.csv")
    parser.add_argument("--emissions", type=Path, default=REPO_ROOT / "outputs" / "emissions" / "original" / "analysis_original.xlsx")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "reproduced_city_features_emissions.xlsx")
    args = parser.parse_args()

    metadata = pd.read_excel(args.metadata, sheet_name="Sheet3")
    descriptors = pd.read_csv(args.descriptors).dropna(axis=1, how="all")
    emissions = pd.read_excel(args.emissions)
    emissions["NO"] = emissions["City"].map(lambda value: int(re.match(r"\d+", str(value)).group()))
    rename = {f"Normalized_{i}OD": f"Normalized carbon emissions_{'OD' if i == 1 else f'{i}OD'}" for i in range(1, 11)}
    emissions = emissions[["NO", *rename]].rename(columns=rename)

    # These two paper descriptors predate the final directed assignment-table
    # representation and are therefore retained from the versioned study table.
    metadata_columns = [
        "NO", "City", "Country", "Continent",
        "Average capacity", "Average speed limit",
    ]
    descriptor_columns = [c for c in descriptors.columns if c != "NO"]
    result = metadata[metadata_columns].merge(descriptors[["NO", *descriptor_columns]], on="NO", how="inner")
    result = result.merge(emissions, on="NO", how="inner").sort_values("NO")
    missing = [f"Normalized carbon emissions_{od}" for od in OD_LEVELS if f"Normalized carbon emissions_{od}" not in result]
    if missing or len(result) != 140:
        raise RuntimeError(f"Incomplete reproduced table: rows={len(result)}, missing={missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(args.output, sheet_name="Sheet3", index=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
