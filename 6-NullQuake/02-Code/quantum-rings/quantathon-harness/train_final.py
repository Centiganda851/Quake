import json
from pathlib import Path

import joblib
import pandas as pd

from ml_common import (
    load_dataset,
    make_model,
)


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def main():
    (
        data,
        X,
        y_log,
        y_seconds,
        groups,
        feature_names,
    ) = load_dataset()

    results = pd.read_csv(
        "benchmark_results.csv"
    ).dropna(subset=["oof_score"])

    results = results.sort_values(
        "oof_score",
        ascending=False,
    )

    best_individual_name = results.iloc[0]["model"]
    best_individual_score = float(
        results.iloc[0]["oof_score"]
    )

    ensemble_path = Path("ensemble_choice.json")

    use_ensemble = False
    ensemble_info = None

    if ensemble_path.exists():
        ensemble_info = json.loads(
            ensemble_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            float(ensemble_info["oof_score"])
            > best_individual_score
        ):
            use_ensemble = True

    if use_ensemble:
        print("Training final ENSEMBLE")
        print(
            "CV score:",
            f"{ensemble_info['oof_score'] * 100:.2f}%"
        )

        models = {}

        for name in ensemble_info["models"]:
            print(f"Training {name} on all {len(X)} rows...")
            model = make_model(name)
            model.fit(X, y_log)
            models[name] = model

        artifact = {
            "kind": "ensemble",
            "models": models,
            "model_names": ensemble_info["models"],
            "weights": ensemble_info["weights"],
            "feature_names": feature_names,
            "target_transform": "log10_seconds",
            "cv_score": float(
                ensemble_info["oof_score"]
            ),
        }

    else:
        print(
            f"Training final individual model: "
            f"{best_individual_name}"
        )
        print(
            "CV score:",
            f"{best_individual_score * 100:.2f}%"
        )

        model = make_model(
            best_individual_name
        )

        model.fit(X, y_log)

        artifact = {
            "kind": "single",
            "model": model,
            "model_name": best_individual_name,
            "feature_names": feature_names,
            "target_transform": "log10_seconds",
            "cv_score": best_individual_score,
        }

    output = (
        ARTIFACTS_DIR
        / "final_model.joblib"
    )

    joblib.dump(
        artifact,
        output,
        compress=3,
    )

    print()
    print(f"Saved: {output}")
    print(
        f"Feature count: "
        f"{len(feature_names)}"
    )


if __name__ == "__main__":
    main()
