# Tree-based weighted ensemble regressor
# It combines: Extra Trees Regressor, 
# Random Forest Regressor, 
# Histogram Gradient Boosting Regressor

import re

CAP_SECONDS = 4 * 60 * 60  # 4-hour timeout cap


class RuntimeModel:
    def __init__(self, artifacts_dir="artifacts"):
        # Load your trained model here, e.g.:
        #   import joblib
        #   self.model = joblib.load(f"{artifacts_dir}/model.pkl")
        self.model = None

    # ------------------------------------------------------------------ #
    # 1) FEATURE PARSER  --  .qasm text  ->  feature dict                 #
    # ------------------------------------------------------------------ #
    def featurize(self, qasm_text: str) -> dict:
        # BASELINE: a few cheap structural features. Design your own.
        n_qubits = 0
        for m in re.finditer(
            r"q(?:u)?(?:reg|bit)\s+\w+\s*\[\s*(\d+)\s*\]",
            qasm_text
        ):
            n_qubits += int(m.group(1))

        lines = [l.strip() for l in qasm_text.splitlines()]

        gate_lines = [
            l
            for l in lines
            if l
            and not l.startswith(
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

        # crude two-qubit-gate count
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
    # 2) MODEL  --  (features, threshold)  ->  predicted seconds          #
    # ------------------------------------------------------------------ #
    def predict(self, features: dict, threshold: int) -> float:
        if self.model is None:
            ...
            import csv
            import math
            from pathlib import Path

            import numpy as np
            import zstandard as zstd

            from sklearn.ensemble import (
                ExtraTreesRegressor,
                RandomForestRegressor,
                HistGradientBoostingRegressor,
            )

            from sklearn.model_selection import GroupShuffleSplit

            base_dir = Path(__file__).resolve().parent
            labels_path = base_dir / "runtime-data.csv"
            circuits_dir = base_dir.parent / "training_circuits"

            training_rows = []

            # ----------------------------------------------------------
            # 1. Read runtime labels
            # ----------------------------------------------------------
            with open(labels_path, newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    filename = row["filename"]
                    thr = int(row["threshold"])

                    if row["status"].lower() == "timeout":
                        runtime_seconds = CAP_SECONDS
                    else:
                        runtime_seconds = float(row["duration_s"])

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

            # ----------------------------------------------------------
            # 2. Featurize every training circuit once
            # ----------------------------------------------------------
            circuit_features = {}

            for filename, _, _ in training_rows:
                if filename in circuit_features:
                    continue

                compressed_path = (
                    circuits_dir / f"{filename}.zst"
                )

                with open(compressed_path, "rb") as f:
                    dctx = zstd.ZstdDecompressor()

                    with dctx.stream_reader(f) as reader:
                        qasm_text = reader.read().decode(
                            "utf-8",
                            errors="replace",
                        )

                circuit_features[filename] = self.featurize(
                    qasm_text
                )

            # ----------------------------------------------------------
            # 3. Turn the existing features into ML inputs
            # ----------------------------------------------------------
            def make_row(f, thr):
                n_qubits = float(f["n_qubits"])
                n_ops = float(f["n_ops"])
                n_2q = float(f["n_2q"])

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

                log_qubits = math.log1p(n_qubits)
                log_ops = math.log1p(n_ops)
                log_2q = math.log1p(n_2q)
                log_threshold = math.log2(thr)

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
                ]

            X_all = []
            y_all = []
            groups = []

            for filename, thr, runtime_seconds in training_rows:
                f = circuit_features[filename]

                X_all.append(
                    make_row(f, thr)
                )

                y_all.append(
                    math.log10(
                        max(
                            runtime_seconds,
                            1e-6,
                        )
                    )
                )

                # This is crucial:
                # all thresholds from the same circuit stay together.
                groups.append(filename)

            X_all = np.asarray(
                X_all,
                dtype=float,
            )

            y_all = np.asarray(
                y_all,
                dtype=float,
            )

            groups = np.asarray(groups)

            # ----------------------------------------------------------
            # 4. Split by CIRCUIT:
            #
            #    2/3 circuits = training
            #    1/3 circuits = validation
            # ----------------------------------------------------------
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

            X_fit = X_all[fit_idx]
            y_fit = y_all[fit_idx]

            X_test = X_all[test_idx]
            y_test = y_all[test_idx]

            print(
                f"Training rows: {len(X_fit)}"
            )

            print(
                f"Held-out rows: {len(X_test)}"
            )

            print(
                f"Training circuits: "
                f"{len(set(groups[fit_idx]))}"
            )

            print(
                f"Held-out circuits: "
                f"{len(set(groups[test_idx]))}"
            )

            # ----------------------------------------------------------
            # 5. Build the three regression models
            # ----------------------------------------------------------
            extra = ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=1,
                max_features=1.0,
                random_state=42,
                n_jobs=-1,
            )

            forest = RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=1,
                max_features=0.9,
                random_state=43,
                n_jobs=-1,
            )

            hist = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                l2_regularization=0.5,
                random_state=44,
            )

            # ----------------------------------------------------------
            # 6. Train ONLY on the 2/3 training circuits
            # ----------------------------------------------------------
            extra.fit(
                X_fit,
                y_fit,
            )

            forest.fit(
                X_fit,
                y_fit,
            )

            hist.fit(
                X_fit,
                y_fit,
            )

            # ----------------------------------------------------------
            # 7. Test on the unseen 1/3 of circuits
            # ----------------------------------------------------------
            extra_test = extra.predict(X_test)
            forest_test = forest.predict(X_test)
            hist_test = hist.predict(X_test)

            blend_test = (
                0.50 * extra_test
                + 0.25 * forest_test
                + 0.25 * hist_test
            )

            actual_seconds = 10 ** y_test
            predicted_seconds = 10 ** blend_test

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

            # Same style of score used by the competition
            held_out_scores = np.maximum(
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

            held_out_score = (
                np.mean(held_out_scores) * 100
            )

            print(
                "\n----------------------------------"
            )

            print(
                f"HELD-OUT SCORE: "
                f"{held_out_score:.2f}%"
            )

            print(
                "----------------------------------\n"
            )

            # Keep the trained models for predictions
            self.model = {
                "extra": extra,
                "forest": forest,
                "hist": hist,
            }

        # --------------------------------------------------------------
        # 8. Predict a circuit given to us by run.py
        # --------------------------------------------------------------
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
            ]],
            dtype=float,
        )

        # --------------------------------------------------------------
        # 9. Get prediction from all 3 regression models
        # --------------------------------------------------------------
        extra_pred = self.model[
            "extra"
        ].predict(X)[0]

        forest_pred = self.model[
            "forest"
        ].predict(X)[0]

        hist_pred = self.model[
            "hist"
        ].predict(X)[0]

        # Weighted ensemble
        predicted_log_seconds = (
            0.50 * extra_pred
            + 0.25 * forest_pred
            + 0.25 * hist_pred
        )

        predicted_seconds = (
            10 ** predicted_log_seconds
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