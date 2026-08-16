"""Compute city-level SHAP values without producing figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.modeling.train_models import FEATURES, OD_LEVELS
from src.paths import PROCESSED_DATA, ensure_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--scaler-dir", type=Path, default=Path("outputs/scalers"))
    parser.add_argument("--od-levels", default=",".join(OD_LEVELS))
    parser.add_argument("--max-models", type=int, default=None)
    args = parser.parse_args()

    data = pd.read_excel(args.data, sheet_name="Sheet3")
    output = ensure_output("shap")
    importance_rows = []
    for od in [v.strip() for v in args.od_levels.split(",") if v.strip()]:
        model_files = sorted(args.model_dir.glob(f"xgboost_{od}_seed*.joblib"))
        if args.max_models:
            model_files = model_files[: args.max_models]
        if not model_files:
            raise FileNotFoundError(f"No XGBoost models found for {od} in {args.model_dir}")
        scaler = joblib.load(args.scaler_dir / f"x_scaler_{od}.joblib")
        raw_features = data[FEATURES].astype(float)
        scaler_input = (
            raw_features
            if hasattr(scaler, "feature_names_in_")
            else raw_features.to_numpy()
        )
        x = pd.DataFrame(scaler.transform(scaler_input), columns=FEATURES)
        values = []
        for model_file in model_files:
            model = joblib.load(model_file)
            values.append(np.asarray(shap.TreeExplainer(model).shap_values(x)))
        mean_values = np.mean(np.stack(values), axis=0)
        city_values = pd.DataFrame(mean_values, columns=FEATURES)
        city_values.insert(0, "City", data["City"].values)
        city_values.insert(0, "NO", data["NO"].values)
        city_values.to_csv(output / f"shap_values_{od}.csv", index=False)
        mean_abs = np.mean(np.abs(mean_values), axis=0)
        total = mean_abs.sum()
        for feature, value in zip(FEATURES, mean_abs):
            importance_rows.append({
                "od_level": od, "feature": feature, "mean_abs_shap": value,
                "importance_share_pct": 100 * value / total,
            })
        print(f"completed SHAP for {od} using {len(model_files)} models")
    pd.DataFrame(importance_rows).to_csv(output / "global_feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
