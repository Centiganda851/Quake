import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold


# ============================================================
# CONFIG
# ============================================================

RUNTIME_FILE = "runtime-data.csv"
MODEL_OUTPUT_FILE = "trained_model.joblib"
OOF_OUTPUT_FILE = "oof_predictions.csv"
EXPERIMENT_OUTPUT_FILE = "experiment_results.csv"

TIMEOUT_SECONDS = 14400.0
N_FOLDS = 5
RANDOM_STATE = 42


# ============================================================
# COMPETITION SCORE
# ============================================================

def competition_score(actual_seconds, predicted_seconds):
    """
    Implements the score formula used by the challenge scorer.

    score = max(
        0,
        1 - abs(log10(predicted / actual)) / 2
    )

    Returns the mean score from 0 to 1.
    """

    actual = np.asarray(actual_seconds, dtype=float)
    predicted = np.asarray(predicted_seconds, dtype=float)

    # Prevent invalid logs.
    actual = np.maximum(actual, 1e-9)
    predicted = np.maximum(predicted, 1e-9)

    individual_scores = np.maximum(
        0.0,
        1.0 - np.abs(np.log10(predicted / actual)) / 2.0,
    )

    return float(np.mean(individual_scores))


# ============================================================
# DATA LOADING
# ============================================================

def load_data(feature_file):
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    runtime_df = pd.read_csv(RUNTIME_FILE)
    feature_df = pd.read_csv(feature_file)

    print(f"Runtime rows : {len(runtime_df)}")
    print(f"Feature rows : {len(feature_df)}")

    if "filename" not in runtime_df.columns:
        raise ValueError("runtime-data.csv must contain a 'filename' column.")

    if "filename" not in feature_df.columns:
        raise ValueError(
            f"{feature_file} must contain a 'filename' column."
        )

    # One row per circuit is expected in feature file.
    duplicate_features = feature_df["filename"].duplicated().sum()

    if duplicate_features:
        raise ValueError(
            f"{feature_file} contains {duplicate_features} duplicate filenames. "
            "Feature file should have one row per circuit."
        )

    # Merge runtime labels with circuit features.
    df = runtime_df.merge(
        feature_df,
        on="filename",
        how="left",
        validate="many_to_one",
    )

    # Check if any circuits failed to match.
    feature_columns_before_cleanup = [
        c for c in feature_df.columns
        if c != "filename"
    ]

    if feature_columns_before_cleanup:
        missing_mask = df[feature_columns_before_cleanup].isna().all(axis=1)

        if missing_mask.any():
            missing_files = df.loc[
                missing_mask,
                "filename"
            ].unique()

            print()
            print("ERROR: Features missing for some circuits:")
            print(missing_files[:20])

            raise ValueError(
                f"Missing features for {len(missing_files)} circuit(s)."
            )

    # --------------------------------------------------------
    # Create the runtime target
    # --------------------------------------------------------

    df["runtime_seconds"] = pd.to_numeric(
        df["duration_s"],
        errors="coerce",
    )

    # Timeout rows count as the timeout cap.
    timeout_mask = (
        df["status"]
        .astype(str)
        .str.lower()
        .eq("timeout")
    )

    df.loc[
        timeout_mask,
        "runtime_seconds"
    ] = TIMEOUT_SECONDS

    # Make sure every row now has a target.
    if df["runtime_seconds"].isna().any():
        problem_rows = df[
            df["runtime_seconds"].isna()
        ][["filename", "threshold", "duration_s", "status"]]

        print(problem_rows.head())

        raise ValueError(
            "Some rows still have no runtime target."
        )

    # Log-transform the target.
    df["target_log10"] = np.log10(
        np.maximum(df["runtime_seconds"], 1e-9)
    )

    # --------------------------------------------------------
    # Automatically discover feature columns
    # --------------------------------------------------------

    ignore_columns = {
        "filename",
        "duration_s",
        "status",
        "shots",
        "backend",
        "precision",
        "runtime_seconds",
        "target_log10",
    }

    candidate_columns = [
        c for c in df.columns
        if c not in ignore_columns
    ]

    # Keep numeric columns only.
    feature_columns = []

    for column in candidate_columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    # threshold should be included automatically because it is numeric.
    if "threshold" not in feature_columns:
        raise ValueError(
            "'threshold' was not detected as a numeric model feature."
        )

    print()
    print("MODEL FEATURES:")

    for feature in feature_columns:
        print(f"  - {feature}")

    print()
    print(f"Number of features: {len(feature_columns)}")

    # Basic missing-value cleanup.
    for column in feature_columns:
        if df[column].isna().any():
            median = df[column].median()

            print(
                f"Filling missing values in {column} "
                f"with median={median}"
            )

            df[column] = df[column].fillna(median)

    return df, feature_columns


# ============================================================
# MODELS
# ============================================================

def make_model(model_name):
    """
    Create a fresh model instance.
    """

    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate(df, feature_columns, model_name):
    print()
    print("=" * 70)
    print(f"CROSS VALIDATION: {model_name}")
    print("=" * 70)

    X = df[feature_columns]
    y_log = df["target_log10"]

    actual_seconds = df["runtime_seconds"].to_numpy()

    groups = df["filename"]

    group_kfold = GroupKFold(
        n_splits=N_FOLDS
    )

    # Each training row gets one prediction from a model
    # that DID NOT train on that circuit.
    oof_pred_seconds = np.zeros(
        len(df),
        dtype=float,
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(
            X,
            y_log,
            groups=groups,
        ),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train_log = y_log.iloc[train_idx]

        model = make_model(model_name)

        model.fit(
            X_train,
            y_train_log,
        )

        pred_log = model.predict(X_val)

        pred_seconds = np.power(
            10.0,
            pred_log,
        )

        # Don't allow nonsense negative/zero values.
        pred_seconds = np.maximum(
            pred_seconds,
            1e-6,
        )

        oof_pred_seconds[val_idx] = pred_seconds

        fold_actual = actual_seconds[val_idx]

        fold_score = competition_score(
            fold_actual,
            pred_seconds,
        )

        fold_scores.append(fold_score)

        print(
            f"Fold {fold}: "
            f"{fold_score * 100:.2f}%"
        )

    overall_score = competition_score(
        actual_seconds,
        oof_pred_seconds,
    )

    print()
    print(
        f"Average fold score: "
        f"{np.mean(fold_scores) * 100:.2f}%"
    )

    print(
        f"Overall OOF score:  "
        f"{overall_score * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save out-of-fold predictions
    # --------------------------------------------------------

    oof_df = pd.DataFrame(
        {
            "filename": df["filename"],
            "threshold": df["threshold"],
            "actual_seconds": actual_seconds,
            "predicted_seconds": oof_pred_seconds,
        }
    )

    oof_df["ratio"] = (
        oof_df["predicted_seconds"]
        / oof_df["actual_seconds"]
    )

    oof_df["absolute_log10_error"] = np.abs(
        np.log10(
            np.maximum(
                oof_df["ratio"],
                1e-9,
            )
        )
    )

    oof_df.to_csv(
        OOF_OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved OOF predictions to "
        f"{OOF_OUTPUT_FILE}"
    )

    return overall_score, oof_df


# ============================================================
# ERROR ANALYSIS
# ============================================================

def show_worst_predictions(oof_df, n=15):
    print()
    print("=" * 70)
    print("WORST VALIDATION PREDICTIONS")
    print("=" * 70)

    worst = (
        oof_df
        .sort_values(
            "absolute_log10_error",
            ascending=False,
        )
        .head(n)
    )

    display_columns = [
        "filename",
        "threshold",
        "actual_seconds",
        "predicted_seconds",
        "ratio",
    ]

    print(
        worst[display_columns]
        .to_string(index=False)
    )


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    df,
    feature_columns,
    model_name,
    cv_score,
):
    print()
    print("=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    X = df[feature_columns]
    y_log = df["target_log10"]

    final_model = make_model(model_name)

    final_model.fit(
        X,
        y_log,
    )

    bundle = {
        "model": final_model,
        "feature_columns": feature_columns,
        "model_name": model_name,
        "cv_score": cv_score,
        "target": "log10(runtime_seconds)",
        "timeout_seconds": TIMEOUT_SECONDS,
    }

    joblib.dump(
        bundle,
        MODEL_OUTPUT_FILE,
    )

    print(
        f"Saved trained model to "
        f"{MODEL_OUTPUT_FILE}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    df,
    feature_columns,
    model_name,
):
    # HistGradientBoosting does not expose feature_importances_
    if model_name == "hist_gradient_boosting":
        return

    model = make_model(model_name)

    model.fit(
        df[feature_columns],
        df["target_log10"],
    )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURE_FILE,
        help=(
            "Feature CSV file. "
            "Default: dummy_features.csv"
        ),
    )

    parser.add_argument(
        "--model",
        default="extra_trees",
        choices=[
            "extra_trees",
            "random_forest",
            "hist_gradient_boosting",
        ],
    )

    args = parser.parse_args()

    feature_file = args.features
    model_name = args.model

    print()
    print("Feature file:", feature_file)
    print("Model:", model_name)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df, feature_columns = load_data(
        feature_file
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    cv_score, oof_df = cross_validate(
        df,
        feature_columns,
        model_name,
    )

    # --------------------------------------------------------
    # Examine failures
    # --------------------------------------------------------

    show_worst_predictions(
        oof_df,
        n=15,
    )

    # --------------------------------------------------------
    # Show feature importance
    # --------------------------------------------------------

    show_feature_importance(
        df,
        feature_columns,
        model_name,
    )

    # --------------------------------------------------------
    # Train final model using all training rows
    # --------------------------------------------------------

    train_final_model(
        df,
        feature_columns,
        model_name,
        cv_score,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"CV score: "
        f"{cv_score * 100:.2f}%"
    )


if __name__ == "__main__":
    main()