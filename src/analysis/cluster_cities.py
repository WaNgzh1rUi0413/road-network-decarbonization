"""Reproduce the two-city-regime hierarchical clustering analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.modeling.train_models import OD_LEVELS
from src.paths import PROCESSED_DATA, ensure_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA)
    parser.add_argument("--clusters", type=int, default=2)
    args = parser.parse_args()
    data = pd.read_excel(args.data, sheet_name="Sheet3")
    columns = [f"Normalized carbon emissions_{od}" for od in OD_LEVELS]
    clustering_features = ["Average capacity", "Average speed limit", "Average degree", "Density"]
    x = StandardScaler().fit_transform(data[clustering_features])
    labels = AgglomerativeClustering(n_clusters=args.clusters, linkage="ward").fit_predict(x)

    increase = data[columns[-1]].to_numpy() - data[columns[0]].to_numpy()
    cluster_increase = {label: float(np.mean(increase[labels == label])) for label in np.unique(labels)}
    ordered = sorted(cluster_increase, key=cluster_increase.get)
    names = {ordered[0]: "demand-resilient", ordered[-1]: "demand-sensitive"}
    result = data[["NO", "City", "Country", "Continent"] + clustering_features + columns].copy()
    result["cluster"] = labels
    result["response_regime"] = [names.get(v, f"cluster-{v}") for v in labels]
    output = ensure_output("clustering")
    result.to_csv(output / "city_response_regimes.csv", index=False)
    pd.DataFrame([{
        "n_clusters": args.clusters,
        "silhouette": silhouette_score(x, labels),
        "calinski_harabasz": calinski_harabasz_score(x, labels),
        "davies_bouldin": davies_bouldin_score(x, labels),
    }]).to_csv(output / "clustering_metrics.csv", index=False)
    print(result["response_regime"].value_counts().to_string())


if __name__ == "__main__":
    main()
