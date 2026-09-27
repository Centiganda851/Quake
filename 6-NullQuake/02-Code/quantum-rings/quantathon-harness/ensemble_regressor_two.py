# 60% XGBoost
# 25% Extra Trees
# 15% LightGBM

import re

CAP_SECONDS = 4 * 60 * 60


class RuntimeModel:
    def __init__(self, artifacts_dir="artifacts"):
        self.model = None

    # ------------------------------------------------------------------ #
    # FEATURE PARSER
    # ------------------------------------------------------------------ #
    def featurize(self, qasm_text: str) -> dict:
        n_qubits = 0

        for m in re.finditer(
            r"q(?:u)?(?:reg|bit)\s+\w+\s*\[\s*(\d+)\s*\]",
            qasm_text,
        ):
            n_qubits += int(m.group(1))

        lines = [
            line.strip()
            for line in qasm_text.splitlines()
        ]

        gate_lines = [
            line
            for line in lines
            if line
            and not line.startswith(
                (
                    "//",
                    "OPENQASM",
                    "include",
                    "qreg",
                    "creg",
                    "qubit",
                    "bit",
                    "gate",
                )
            )
        ]

        n_ops = len(gate_lines)

        n_2q = len(
            re.findall(
                r"\b(cx|cz|cy|ch|swap|iswap|rzz|rxx|ryy|cp|crx|cry|crz|ecr)\b",
                qasm_text,
            )
        )

        return {
            "n_qubits": n_qubits,
            "n_ops": n_ops,
            "n_2q": n_2q,
        }

    # ------------------------------------------------------------------ #
    # MODEL
    # ------------------------------------------------------------------ #
    def predict(self, features: dict, threshold: int) -> float:

        if self.model is None:
            import csv
            import math
            from pathlib import Path

            import numpy as np
            import zstandard as zstd

            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.model_selection import GroupShuffleSplit

            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor

            base_dir = Path(__file__).resolve().parent

            labels_path = (
                base_dir / "runtime-data.csv"
            )

            circuits_dir = (
                base_dir.parent / "training_circuits"
            )

            # ---------------------------------------------------------- #
            # 1. Read runtime labels
            # ---------------------------------------------------------- #
            training_rows = []

            with open(labels_path, newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    filename = row["filename"]
                    thr = int(row["threshold"])

                    if row["status"].lower() == "timeout":
                        runtime_seconds = CAP_SECONDS
                    else:
                        runtime_seconds = float(
                            row["duration_s"]
                        )

                    runtime_seconds = min(
                        runtime_seconds,
                        CAP_SECONDS,
                    )

                    training_rows.append(
                        (
                            filename,
                            thr,
                            runtime_seconds,
                        )
                    )

            # ---------------------------------------------------------- #
            # 2. Featurize every training circuit once
            # ---------------------------------------------------------- #
            circuit_features = {}

            for filename, _, _ in training_rows:

                if filename in circuit_features:
                    continue

                compressed_path = (
                    circuits_dir
                    / f"{filename}.zst"
                )

                with open(
                    compressed_path,
                    "rb",
                ) as f:

                    dctx = (
                        zstd.ZstdDecompressor()
                    )

                    with dctx.stream_reader(
                        f
                    ) as reader:

                        qasm_text = (
                            reader.read()
                            .decode(
                                "utf-8",
                                errors="replace",
                            )
                        )

                circuit_features[
                    filename
                ] = self.featurize(
                    qasm_text
                )

            # ---------------------------------------------------------- #
            # 3. Create ML features
            # ---------------------------------------------------------- #
            def make_row(f, thr):

                n_qubits = float(
                    f["n_qubits"]
                )

                n_ops = float(
                    f["n_ops"]
                )

                n_2q = float(
                    f["n_2q"]
                )

                two_q_ratio = (
                    n_2q / n_ops
                    if n_ops > 0
                    else 0.0
                )

                ops_per_qubit = (
                    n_ops / n_qubits
                    if n_qubits > 0
                    else 0.0
                )

                two_q_per_qubit = (
                    n_2q / n_qubits
                    if n_qubits > 0
                    else 0.0
                )

                log_qubits = math.log1p(
                    n_qubits
                )

                log_ops = math.log1p(
                    n_ops
                )

                log_2q = math.log1p(
                    n_2q
                )

                log_threshold = math.log2(
                    thr
                )

                return [
                    n_qubits,
                    n_ops,
                    n_2q,

                    two_q_ratio,
                    ops_per_qubit,
                    two_q_per_qubit,

                    log_qubits,
                    log_ops,
                    log_2q,

                    float(thr),
                    log_threshold,

                    # interaction features
                    n_qubits * log_ops,
                    n_qubits * two_q_ratio,
                    log_ops * log_threshold,
                    n_qubits * log_threshold,
                    n_2q * log_threshold,

                    # extra nonlinear features
                    n_qubits * n_qubits,
                    log_ops * log_ops,
                    two_q_ratio * log_threshold,
                ]

            # ---------------------------------------------------------- #
            # 4. Build training dataset
            # ---------------------------------------------------------- #
            X_all = []
            y_all = []
            groups = []

            for (
                filename,
                thr,
                runtime_seconds,
            ) in training_rows:

                f = circuit_features[
                    filename
                ]

                X_all.append(
                    make_row(
                        f,
                        thr,
                    )
                )

                y_all.append(
                    math.log10(
                        max(
                            runtime_seconds,
                            1e-6,
                        )
                    )
                )

                groups.append(
                    filename
                )

            X_all = np.asarray(
                X_all,
                dtype=float,
            )

            y_all = np.asarray(
                y_all,
                dtype=float,
            )

            groups = np.asarray(
                groups
            )

            # ---------------------------------------------------------- #
            # 5. 2/3 train, 1/3 held out by CIRCUIT
            # ---------------------------------------------------------- #
            splitter = GroupShuffleSplit(
                n_splits=1,
                train_size=2 / 3,
                random_state=42,
            )

            fit_idx, test_idx = next(
                splitter.split(
                    X_all,
                    y_all,
                    groups=groups,
                )
            )

            X_fit = X_all[
                fit_idx
            ]

            y_fit = y_all[
                fit_idx
            ]

            X_test = X_all[
                test_idx
            ]

            y_test = y_all[
                test_idx
            ]

            print(
                f"Training rows: "
                f"{len(X_fit)}"
            )

            print(
                f"Held-out rows: "
                f"{len(X_test)}"
            )

            print(
                "Training circuits: "
                f"{len(set(groups[fit_idx]))}"
            )

            print(
                "Held-out circuits: "
                f"{len(set(groups[test_idx]))}"
            )

            # ---------------------------------------------------------- #
            # 6. XGBoost
            # ---------------------------------------------------------- #
            xgb = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=2,
                subsample=0.90,
                colsample_bytree=0.90,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )

            # ---------------------------------------------------------- #
            # 7. Extra Trees
            # ---------------------------------------------------------- #
            extra = ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=1,
                max_features=1.0,
                random_state=43,
                n_jobs=-1,
            )

            # ---------------------------------------------------------- #
            # 8. LightGBM
            # ---------------------------------------------------------- #
            lgbm = LGBMRegressor(
                objective="regression",
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=10,
                subsample=0.90,
                colsample_bytree=0.90,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=44,
                n_jobs=-1,
                verbosity=-1,
            )

            # ---------------------------------------------------------- #
            # 9. Train all 3 models
            # ---------------------------------------------------------- #
            print("\nTraining XGBoost...")

            xgb.fit(
                X_fit,
                y_fit,
            )

            print("Training Extra Trees...")

            extra.fit(
                X_fit,
                y_fit,
            )

            print("Training LightGBM...")

            lgbm.fit(
                X_fit,
                y_fit,
            )

            # ---------------------------------------------------------- #
            # 10. Individual held-out predictions
            # ---------------------------------------------------------- #
            xgb_test = (
                xgb.predict(
                    X_test
                )
            )

            extra_test = (
                extra.predict(
                    X_test
                )
            )

            lgbm_test = (
                lgbm.predict(
                    X_test
                )
            )

            # ---------------------------------------------------------- #
            # 11. Weighted ensemble:
            #
            #     60% XGBoost
            #     25% Extra Trees
            #     15% LightGBM
            #
            # Weighting happens in log10(runtime) space.
            # ---------------------------------------------------------- #
            ensemble_test = (
                0.60 * xgb_test
                + 0.25 * extra_test
                + 0.15 * lgbm_test
            )

            # ---------------------------------------------------------- #
            # Helper for scoring
            # ---------------------------------------------------------- #
            def competition_score(
                actual_log,
                predicted_log,
            ):
                actual_seconds = (
                    10 ** actual_log
                )

                predicted_seconds = (
                    10 ** predicted_log
                )

                actual_seconds = np.minimum(
                    actual_seconds,
                    CAP_SECONDS,
                )

                predicted_seconds = np.minimum(
                    predicted_seconds,
                    CAP_SECONDS,
                )

                predicted_seconds = np.maximum(
                    predicted_seconds,
                    1e-6,
                )

                scores = np.maximum(
                    0.0,
                    1.0
                    - (
                        np.abs(
                            np.log10(
                                predicted_seconds
                                / actual_seconds
                            )
                        )
                        / 2.0
                    ),
                )

                return (
                    np.mean(scores)
                    * 100
                )

            # ---------------------------------------------------------- #
            # 12. Print individual + ensemble scores
            # ---------------------------------------------------------- #
            xgb_score = competition_score(
                y_test,
                xgb_test,
            )

            extra_score = competition_score(
                y_test,
                extra_test,
            )

            lgbm_score = competition_score(
                y_test,
                lgbm_test,
            )

            ensemble_score = competition_score(
                y_test,
                ensemble_test,
            )

            print()
            print(
                "=========================================="
            )

            print(
                f"XGBoost held-out score:     "
                f"{xgb_score:.2f}%"
            )

            print(
                f"Extra Trees held-out score: "
                f"{extra_score:.2f}%"
            )

            print(
                f"LightGBM held-out score:    "
                f"{lgbm_score:.2f}%"
            )

            print(
                "------------------------------------------"
            )

            print(
                f"ENSEMBLE HELD-OUT SCORE:    "
                f"{ensemble_score:.2f}%"
            )

            print(
                "Weights: "
                "60% XGBoost / "
                "25% Extra Trees / "
                "15% LightGBM"
            )

            print(
                "=========================================="
            )
            print()

            # Keep trained models
            self.model = {
                "xgb": xgb,
                "extra": extra,
                "lgbm": lgbm,
            }

        # -------------------------------------------------------------- #
        # 13. Predict new circuit
        # -------------------------------------------------------------- #
        import math
        import numpy as np

        n_qubits = float(
            features["n_qubits"]
        )

        n_ops = float(
            features["n_ops"]
        )

        n_2q = float(
            features["n_2q"]
        )

        two_q_ratio = (
            n_2q / n_ops
            if n_ops > 0
            else 0.0
        )

        ops_per_qubit = (
            n_ops / n_qubits
            if n_qubits > 0
            else 0.0
        )

        two_q_per_qubit = (
            n_2q / n_qubits
            if n_qubits > 0
            else 0.0
        )

        log_qubits = math.log1p(
            n_qubits
        )

        log_ops = math.log1p(
            n_ops
        )

        log_2q = math.log1p(
            n_2q
        )

        log_threshold = math.log2(
            threshold
        )

        X = np.asarray(
            [[
                n_qubits,
                n_ops,
                n_2q,

                two_q_ratio,
                ops_per_qubit,
                two_q_per_qubit,

                log_qubits,
                log_ops,
                log_2q,

                float(threshold),
                log_threshold,

                n_qubits * log_ops,
                n_qubits * two_q_ratio,
                log_ops * log_threshold,
                n_qubits * log_threshold,
                n_2q * log_threshold,

                n_qubits * n_qubits,
                log_ops * log_ops,
                two_q_ratio * log_threshold,
            ]],
            dtype=float,
        )

        # -------------------------------------------------------------- #
        # 14. Each model predicts log10(seconds)
        # -------------------------------------------------------------- #
        xgb_pred = (
            self.model[
                "xgb"
            ].predict(X)[0]
        )

        extra_pred = (
            self.model[
                "extra"
            ].predict(X)[0]
        )

        lgbm_pred = (
            self.model[
                "lgbm"
            ].predict(X)[0]
        )

        # -------------------------------------------------------------- #
        # 15. Weighted ensemble in log space
        # -------------------------------------------------------------- #
        predicted_log_seconds = (
            0.60 * xgb_pred
            + 0.25 * extra_pred
            + 0.15 * lgbm_pred
        )

        predicted_seconds = (
            10
            ** predicted_log_seconds
        )

        return float(
            min(
                max(
                    predicted_seconds,
                    1e-6,
                ),
                CAP_SECONDS,
            )
        )