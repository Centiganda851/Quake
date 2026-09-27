from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CAP_SECONDS = 14400.0

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent

FEATURES_PATH = REPO_ROOT / "prosanta" / "circuit_features_flattened.csv"
LABELS_PATH = BASE_DIR.parent / "runtime-data.csv"


def competition_score(actual_seconds, predicted_seconds):
    actual = np.asarray(actual_seconds, dtype=float)
    pred = np.asarray(predicted_seconds, dtype=float)

    actual = np.clip(actual, 1e-9, CAP_SECONDS)
    pred = np.clip(pred, 1e-9, CAP_SECONDS)

    scores = np.maximum(
        0.0,
        1.0 - np.abs(np.log10(pred / actual)) / 2.0,
    )

    return float(np.mean(scores))


def load_dataset():
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)

    if features["filename"].duplicated().any():
        raise ValueError("Feature CSV contains duplicate filenames.")

    feature_columns = [
        col
        for col in features.columns
        if col != "filename"
    ]

    merged = labels.merge(
        features,
        on="filename",
        how="left",
        validate="many_to_one",
    )

    missing = merged[feature_columns].isna().any(axis=1)

    if missing.any():
        bad = merged.loc[missing, "filename"].unique().tolist()
        raise ValueError(
            f"Missing feature rows for {len(bad)} circuits: {bad[:10]}"
        )

    runtime_seconds = np.where(
        merged["status"].str.lower().eq("timeout"),
        CAP_SECONDS,
        pd.to_numeric(merged["duration_s"], errors="coerce"),
    )

    runtime_seconds = np.asarray(runtime_seconds, dtype=float)
    runtime_seconds = np.nan_to_num(
        runtime_seconds,
        nan=CAP_SECONDS,
        posinf=CAP_SECONDS,
        neginf=1e-9,
    )
    runtime_seconds = np.clip(runtime_seconds, 1e-9, CAP_SECONDS)

    merged["runtime_seconds"] = runtime_seconds
    merged["log_threshold"] = np.log2(
        merged["threshold"].astype(float)
    )

    model_feature_names = feature_columns + [
        "threshold",
        "log_threshold",
    ]

    X = (
        merged[model_feature_names]
        .astype(float)
        .to_numpy()
    )

    y_seconds = merged["runtime_seconds"].to_numpy(dtype=float)
    y_log = np.log10(y_seconds)

    groups = merged["filename"].to_numpy()

    return (
        merged,
        X,
        y_log,
        y_seconds,
        groups,
        model_feature_names,
    )


def make_model(name):
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )

    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=1,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )

    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.5,
            random_state=42,
        )

    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            loss="huber",
            random_state=42,
        )

    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=700,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    if name == "svr":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVR(
                        kernel="rbf",
                        C=10.0,
                        epsilon=0.05,
                        gamma="scale",
                    ),
                ),
            ]
        )

    if name == "knn":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=8,
                        weights="distance",
                        p=2,
                    ),
                ),
            ]
        )

    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=0.001,
                        learning_rate_init=0.001,
                        max_iter=1500,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = [
    "extra_trees",
    "random_forest",
    "hist_gradient_boosting",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "svr",
    "knn",
    "mlp",
]


def feature_dict_to_row(features, threshold, feature_names):
    values = []

    for name in feature_names:
        if name == "threshold":
            value = float(threshold)

        elif name == "log_threshold":
            value = math.log2(float(threshold))

        else:
            value = float(features.get(name, 0.0))

        if not math.isfinite(value):
            value = 0.0

        values.append(value)

    return values
