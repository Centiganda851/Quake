import time

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from ml_common import (
    CAP_SECONDS,
    MODEL_NAMES,
    competition_score,
    load_dataset,
    make_model,
)


def main():
    (
        data,
        X,
        y_log,
        y_seconds,
        groups,
        feature_names,
    ) = load_dataset()

    print("=" * 70)
    print("REAL FEATURE MODEL BENCHMARK")
    print("=" * 70)
    print(f"Rows: {len(data)}")
    print(f"Circuits: {data['filename'].nunique()}")
    print(f"Input features: {len(feature_names)}")
    print(f"Feature matrix: {X.shape}")
    print()

    splitter = GroupKFold(n_splits=5)

    results = []
    oof_df = data[
        ["filename", "threshold", "runtime_seconds"]
    ].copy()

    for model_name in MODEL_NAMES:
        print()
        print("=" * 70)
        print(model_name.upper())
        print("=" * 70)

        oof_log = np.full(len(data), np.nan, dtype=float)
        fold_scores = []
        fold_train_times = []
        fold_predict_times = []

        try:
            for fold, (train_idx, val_idx) in enumerate(
                splitter.split(X, y_log, groups=groups),
                start=1,
            ):
                model = make_model(model_name)

                start = time.perf_counter()
                model.fit(X[train_idx], y_log[train_idx])
                train_time = time.perf_counter() - start

                start = time.perf_counter()
                pred_log = model.predict(X[val_idx])
                predict_time = time.perf_counter() - start

                oof_log[val_idx] = pred_log

                pred_seconds = np.clip(
                    np.power(10.0, pred_log),
                    1e-9,
                    CAP_SECONDS,
                )

                fold_score = competition_score(
                    y_seconds[val_idx],
                    pred_seconds,
                )

                fold_scores.append(fold_score)
                fold_train_times.append(train_time)
                fold_predict_times.append(predict_time)

                print(
                    f"Fold {fold}: "
                    f"{fold_score * 100:.2f}%  "
                    f"train={train_time:.2f}s  "
                    f"predict={predict_time:.4f}s"
                )

            if np.isnan(oof_log).any():
                raise RuntimeError(
                    f"{model_name} did not produce all OOF predictions."
                )

            oof_seconds = np.clip(
                np.power(10.0, oof_log),
                1e-9,
                CAP_SECONDS,
            )

            overall_score = competition_score(
                y_seconds,
                oof_seconds,
            )

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))

            print(
                f"\n{model_name}: "
                f"OOF={overall_score * 100:.2f}% | "
                f"fold mean={mean_score * 100:.2f}% "
                f"+/- {std_score * 100:.2f}%"
            )

            results.append(
                {
                    "model": model_name,
                    "oof_score": overall_score,
                    "fold_mean": mean_score,
                    "fold_std": std_score,
                    "mean_train_seconds": float(
                        np.mean(fold_train_times)
                    ),
                    "mean_predict_seconds": float(
                        np.mean(fold_predict_times)
                    ),
                }
            )

            oof_df[f"{model_name}_log_pred"] = oof_log

        except Exception as exc:
            print(f"\nFAILED: {model_name}")
            print(repr(exc))

            results.append(
                {
                    "model": model_name,
                    "oof_score": np.nan,
                    "fold_mean": np.nan,
                    "fold_std": np.nan,
                    "mean_train_seconds": np.nan,
                    "mean_predict_seconds": np.nan,
                }
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "oof_score",
        ascending=False,
        na_position="last",
    )

    results_df.to_csv(
        "benchmark_results.csv",
        index=False,
    )

    oof_df.to_csv(
        "oof_predictions_real_features.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL INDIVIDUAL MODEL RANKING")
    print("=" * 70)

    printable = results_df.copy()

    printable["oof_score_percent"] = (
        printable["oof_score"] * 100.0
    )
    printable["fold_std_percent"] = (
        printable["fold_std"] * 100.0
    )

    print(
        printable[
            [
                "model",
                "oof_score_percent",
                "fold_std_percent",
                "mean_train_seconds",
                "mean_predict_seconds",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print("  benchmark_results.csv")
    print("  oof_predictions_real_features.csv")


if __name__ == "__main__":
    main()
