import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold

from lightgbm import LGBMRegressor


CAP_SECONDS = 14400.0

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent

FEATURES_JSON = (
    REPO_ROOT
    / "prosanta"
    / "circuit_features_532.json"
)

LABELS_CSV = (
    HARNESS_DIR.parent
    / "runtime-data.csv"
)


def competition_score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    actual = np.clip(actual, 1e-9, CAP_SECONDS)
    predicted = np.clip(predicted, 1e-9, CAP_SECONDS)

    scores = np.maximum(
        0.0,
        1.0 - np.abs(
            np.log10(predicted / actual)
        ) / 2.0,
    )

    return float(np.mean(scores))


def flatten_circuit_features(filename, features):
    row = {
        "filename": filename,
    }

    for key, value in features.items():
        if key == "gate_counts":
            continue

        if isinstance(value, (int, float)):
            row[key] = value

    gate_counts = features.get(
        "gate_counts",
        {},
    )

    for gate_name, count in gate_counts.items():
        row[f"gate_count_{gate_name}"] = count

    return row


def load_new_features():
    with FEATURES_JSON.open(
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(f)

    rows = []

    for filename, features in raw.items():
        rows.append(
            flatten_circuit_features(
                filename,
                features,
            )
        )

    df = pd.DataFrame(rows)

    # Any gate that does not occur in a circuit should count as 0.
    df = df.fillna(0.0)

    return df


def make_extra_trees():
    return ExtraTreesRegressor(
        n_estimators=600,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )


def make_lightgbm():
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


def make_histgb():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=0.5,
        random_state=42,
    )


def main():
    features = load_new_features()
    labels = pd.read_csv(LABELS_CSV)

    print("=" * 70)
    print("NEW FEATURE SET")
    print("=" * 70)

    print(
        "Circuits in feature file:",
        len(features),
    )

    print(
        "Raw columns including filename:",
        len(features.columns),
    )

    feature_columns = [
        col
        for col in features.columns
        if col != "filename"
    ]

    print(
        "Circuit feature columns:",
        len(feature_columns),
    )

    print()
    print("First feature columns:")
    print(
        feature_columns[:20]
    )

    if features["filename"].duplicated().any():
        raise RuntimeError(
            "Duplicate filenames found."
        )

    merged = labels.merge(
        features,
        on="filename",
        how="left",
        validate="many_to_one",
    )

    missing = merged[
        feature_columns
    ].isna().any(axis=1)

    if missing.any():
        bad = merged.loc[
            missing,
            "filename",
        ].unique()

        raise RuntimeError(
            f"Missing features for: {bad[:10]}"
        )

    # Timeout rows count as exactly 14400 seconds.
    runtime_seconds = np.where(
        merged["status"]
        .astype(str)
        .str.lower()
        .eq("timeout"),
        CAP_SECONDS,
        pd.to_numeric(
            merged["duration_s"],
            errors="coerce",
        ),
    )

    runtime_seconds = np.asarray(
        runtime_seconds,
        dtype=float,
    )

    runtime_seconds = np.nan_to_num(
        runtime_seconds,
        nan=CAP_SECONDS,
        posinf=CAP_SECONDS,
        neginf=1e-9,
    )

    runtime_seconds = np.clip(
        runtime_seconds,
        1e-9,
        CAP_SECONDS,
    )

    merged["runtime_seconds"] = (
        runtime_seconds
    )

    # Threshold is known at prediction time.
    merged["threshold"] = (
        merged["threshold"]
        .astype(float)
    )

    merged["log_threshold"] = np.log2(
        merged["threshold"]
    )

    model_feature_names = (
        feature_columns
        + [
            "threshold",
            "log_threshold",
        ]
    )

    print()
    print(
        "FINAL MODEL INPUT COUNT:",
        len(model_feature_names),
    )

    X = (
        merged[model_feature_names]
        .astype(float)
        .to_numpy()
    )

    y_seconds = (
        merged["runtime_seconds"]
        .to_numpy(dtype=float)
    )

    y_log = np.log10(
        y_seconds
    )

    groups = (
        merged["filename"]
        .to_numpy()
    )

    thresholds = (
        merged["threshold"]
        .to_numpy(dtype=int)
    )

    print(
        "Training rows:",
        len(merged),
    )

    print(
        "Unique circuits:",
        merged["filename"].nunique(),
    )

    print(
        "X shape:",
        X.shape,
    )

    print()
    print("=" * 70)
    print("5-FOLD GROUPED ENSEMBLE TEST")
    print("=" * 70)

    splitter = GroupKFold(
        n_splits=5
    )

    oof_log = np.full(
        len(merged),
        np.nan,
        dtype=float,
    )

    fold_scores = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            X,
            y_log,
            groups=groups,
        ),
        start=1,
    ):
        extra_trees = (
            make_extra_trees()
        )

        lightgbm = (
            make_lightgbm()
        )

        histgb = (
            make_histgb()
        )

        extra_trees.fit(
            X[train_idx],
            y_log[train_idx],
        )

        lightgbm.fit(
            X[train_idx],
            y_log[train_idx],
        )

        histgb.fit(
            X[train_idx],
            y_log[train_idx],
        )

        et_pred = extra_trees.predict(
            X[test_idx]
        )

        lgb_pred = lightgbm.predict(
            X[test_idx]
        )

        hist_pred = histgb.predict(
            X[test_idx]
        )

        # Same final ensemble weights
        # as your current model.
        ensemble_pred_log = (
            0.70 * et_pred
            + 0.20 * lgb_pred
            + 0.10 * hist_pred
        )

        oof_log[
            test_idx
        ] = ensemble_pred_log

        pred_seconds = np.clip(
            10 ** ensemble_pred_log,
            1e-9,
            CAP_SECONDS,
        )

        score = competition_score(
            y_seconds[test_idx],
            pred_seconds,
        )

        fold_scores.append(score)

        print(
            f"Fold {fold}: "
            f"{score * 100:.2f}%"
        )

    if np.isnan(oof_log).any():
        raise RuntimeError(
            "Some OOF rows were not predicted."
        )

    final_pred_seconds = np.clip(
        10 ** oof_log,
        1e-9,
        CAP_SECONDS,
    )

    overall_score = competition_score(
        y_seconds,
        final_pred_seconds,
    )

    print()
    print("=" * 70)
    print("NEW-FEATURE ENSEMBLE RESULTS")
    print("=" * 70)

    print(
        f"Overall Ensemble OOF: "
        f"{overall_score * 100:.2f}%"
    )

    print(
        f"Fold mean: "
        f"{np.mean(fold_scores) * 100:.2f}%"
    )

    print(
        f"Fold std: "
        f"{np.std(fold_scores) * 100:.2f}%"
    )

    print()
    print("BY THRESHOLD")

    for threshold in [
        16,
        64,
        512,
    ]:
        mask = (
            thresholds
            == threshold
        )

        score = competition_score(
            y_seconds[mask],
            final_pred_seconds[mask],
        )

        print(
            f"Threshold {threshold}: "
            f"{score * 100:.2f}% "
            f"({mask.sum()} rows)"
        )

    print()
    print("=" * 70)
    print("OLD ENSEMBLE REFERENCE")
    print("=" * 70)

    print(
        "Old 94-feature ensemble: "
        "89.21% OOF"
    )

    print()
    print(
        "If the new score is > 89.21%, "
        "the new feature set is an improvement."
    )


if __name__ == "__main__":
    main()
