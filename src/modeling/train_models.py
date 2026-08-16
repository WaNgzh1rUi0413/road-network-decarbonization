"""Train the four regressors and voting ensemble reported in the paper.

The defaults reproduce the paper protocol: ten OD-demand levels, 50 random
train/test splits, min-max scaling, 30-iteration randomized search, and
leave-one-out cross-validation on each training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from src.paths import PROCESSED_DATA, ensure_output


FEATURES = [
    "Average degree", "Clustering coefficient", "Pearson degree correlation",
    "Efficiency", "Transitivity", "Density", "Characteristic path length",
    "Average node strength", "Average speed limit", "Average capacity",
]
OD_LEVELS = ["OD"] + [f"{i}OD" for i in range(2, 11)]
PAPER_RANDOM_STATES = [
    650, 680, 16, 20, 24, 32, 35, 42, 46, 77, 121, 200, 211, 213, 225,
    234, 245, 300, 312, 330, 350, 360, 380, 381, 800, 1900, 1901, 1902,
    1903, 1904, 1905, 2000, 2002, 2003, 2200, 3000, 3001, 3002, 3003,
    3004, 3005, 4000, 4001, 4002, 5000, 900, 901, 902, 903, 904,
]


def model_specs(seed: int, estimator_jobs: int | None = None):
    catboost_options = {
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    if estimator_jobs is not None:
        catboost_options["thread_count"] = estimator_jobs
    return {
        "random_forest": (
            RandomForestRegressor(random_state=seed),
            {
                "n_estimators": randint(20, 80), "max_depth": randint(3, 6),
                "min_samples_split": randint(8, 20), "min_samples_leaf": randint(4, 10),
                "max_features": uniform(0.5, 0.4),
            },
        ),
        "xgboost": (
            XGBRegressor(random_state=seed, n_jobs=estimator_jobs),
            {
                "n_estimators": randint(20, 80), "learning_rate": uniform(0.01, 0.09),
                "max_depth": randint(2, 4), "min_child_weight": randint(3, 10),
                "subsample": uniform(0.6, 0.3), "colsample_bytree": uniform(0.6, 0.3),
                "reg_alpha": uniform(0.01, 0.5), "reg_lambda": uniform(0.5, 2.0),
            },
        ),
        "lightgbm": (
            LGBMRegressor(random_state=seed, verbosity=-1, n_jobs=estimator_jobs),
            {
                "n_estimators": randint(20, 80), "learning_rate": uniform(0.01, 0.09),
                "max_depth": randint(2, 4), "num_leaves": randint(10, 25),
                "min_child_samples": randint(8, 20), "subsample": uniform(0.6, 0.3),
                "colsample_bytree": uniform(0.6, 0.3), "reg_alpha": uniform(0.01, 0.5),
                "reg_lambda": uniform(0.5, 2.0),
            },
        ),
        "catboost": (
            CatBoostRegressor(**catboost_options),
            {
                "iterations": randint(20, 80), "learning_rate": uniform(0.01, 0.09),
                "depth": randint(2, 4), "l2_leaf_reg": uniform(2.0, 5.0),
                "subsample": uniform(0.6, 0.3),
            },
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def run(
    data_path: Path,
    od_levels: list[str],
    seeds: list[int],
    n_iter: int,
    n_jobs: int,
    estimator_jobs: int | None,
) -> None:
    data = pd.read_excel(data_path, sheet_name="Sheet3")
    model_dir = ensure_output("models")
    scaler_dir = ensure_output("scalers")
    result_dir = ensure_output("modeling")
    rows: list[dict] = []

    for od in od_levels:
        target = f"Normalized carbon emissions_{od}"
        x_raw = data[FEATURES].astype(float)
        y_raw = data[[target]].astype(float)
        x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
        x = pd.DataFrame(
            x_scaler.fit_transform(x_raw),
            columns=FEATURES,
            index=data.index,
        )
        y = y_scaler.fit_transform(y_raw).ravel()
        joblib.dump(x_scaler, scaler_dir / f"x_scaler_{od}.joblib")
        joblib.dump(y_scaler, scaler_dir / f"y_scaler_{od}.joblib")

        for seed in seeds:
            indices = np.arange(len(data))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
            fitted = {}
            cv_rmse = {}
            for name, (estimator, space) in model_specs(seed, estimator_jobs).items():
                search = RandomizedSearchCV(
                    estimator, space, n_iter=n_iter, cv=LeaveOneOut(),
                    scoring="neg_root_mean_squared_error", random_state=seed,
                    n_jobs=n_jobs, refit=True,
                )
                search.fit(x.iloc[train_idx], y[train_idx])
                fitted[name] = search.best_estimator_
                cv_rmse[name] = -float(search.best_score_)
                joblib.dump(search.best_estimator_, model_dir / f"{name}_{od}_seed{seed}.joblib")

            best = sorted(cv_rmse, key=cv_rmse.get)[:3]
            ensemble = VotingRegressor([(name, fitted[name]) for name in best]).fit(
                x.iloc[train_idx], y[train_idx]
            )
            fitted["voting_ensemble"] = ensemble
            joblib.dump(ensemble, model_dir / f"voting_ensemble_{od}_seed{seed}.joblib")

            for name, model in fitted.items():
                pred_scaled = model.predict(x.iloc[test_idx]).reshape(-1, 1)
                pred = y_scaler.inverse_transform(pred_scaled).ravel()
                score = metrics(y_raw.iloc[test_idx].to_numpy().ravel(), pred)
                rows.append({"od_level": od, "seed": seed, "model": name, **score})
            print(f"completed od={od}, seed={seed}")

    pd.DataFrame(rows).to_csv(result_dir / "test_metrics.csv", index=False)
    (result_dir / "protocol.json").write_text(
        json.dumps({"features": FEATURES, "od_levels": od_levels, "seeds": seeds, "n_iter": n_iter}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA)
    parser.add_argument("--od-levels", default=",".join(OD_LEVELS))
    parser.add_argument("--seeds", default=",".join(map(str, PAPER_RANDOM_STATES)))
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--estimator-jobs",
        type=int,
        default=0,
        help="Threads per estimator; 0 preserves the paper's library defaults.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run one OD level, one seed and two search draws.")
    args = parser.parse_args()
    ods = [v.strip() for v in args.od_levels.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    if args.smoke_test:
        ods, seeds, args.n_iter, args.n_jobs, args.estimator_jobs = ods[:1], seeds[:1], 2, 1, 1
    estimator_jobs = args.estimator_jobs or None
    run(args.data, ods, seeds, args.n_iter, args.n_jobs, estimator_jobs)


if __name__ == "__main__":
    main()
