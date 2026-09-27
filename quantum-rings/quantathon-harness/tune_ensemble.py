import json

import numpy as np
import pandas as pd

from ml_common import CAP_SECONDS, competition_score


def main():
    results = pd.read_csv("benchmark_results.csv")
    oof = pd.read_csv("oof_predictions_real_features.csv")

    valid = results.dropna(subset=["oof_score"]).sort_values(
        "oof_score",
        ascending=False,
    )

    if len(valid) < 3:
        raise RuntimeError(
            "Need at least 3 successful models before ensemble tuning."
        )

    top3 = valid["model"].head(3).tolist()

    print("Top 3 models:")
    for model in top3:
        score = float(
            valid.loc[
                valid["model"] == model,
                "oof_score",
            ].iloc[0]
        )
        print(f"  {model}: {score * 100:.2f}%")

    actual = oof["runtime_seconds"].to_numpy(dtype=float)

    pred_logs = {
        name: oof[f"{name}_log_pred"].to_numpy(dtype=float)
        for name in top3
    }

    best = None

    # 0.05 increments, weights sum exactly to 1.
    for a_int in range(21):
        for b_int in range(21 - a_int):
            c_int = 20 - a_int - b_int

            a = a_int / 20.0
            b = b_int / 20.0
            c = c_int / 20.0

            blended_log = (
                a * pred_logs[top3[0]]
                + b * pred_logs[top3[1]]
                + c * pred_logs[top3[2]]
            )

            pred_seconds = np.clip(
                np.power(10.0, blended_log),
                1e-9,
                CAP_SECONDS,
            )

            score = competition_score(
                actual,
                pred_seconds,
            )

            if best is None or score > best["oof_score"]:
                best = {
                    "models": top3,
                    "weights": [a, b, c],
                    "oof_score": float(score),
                }

    best_individual = valid.iloc[0]

    print()
    print("=" * 70)
    print("BEST INDIVIDUAL")
    print("=" * 70)
    print(
        f"{best_individual['model']}: "
        f"{best_individual['oof_score'] * 100:.2f}%"
    )

    print()
    print("=" * 70)
    print("BEST 3-MODEL ENSEMBLE")
    print("=" * 70)

    for name, weight in zip(
        best["models"],
        best["weights"],
    ):
        print(f"{name}: {weight:.2f}")

    print(
        f"OOF score: "
        f"{best['oof_score'] * 100:.2f}%"
    )

    with open(
        "ensemble_choice.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best, f, indent=2)

    print()
    print("Saved ensemble_choice.json")


if __name__ == "__main__":
    main()
