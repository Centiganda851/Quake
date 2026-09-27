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
    # HISTOGRAM GRADIENT BOOSTING REGRESSOR
    # ------------------------------------------------------------------ #
    def predict(self, features: dict, threshold: int) -> float:

        if self.model is None:
            import csv
            import math
            from pathlib import Path

            import numpy as np
            import zstandard as zstd

            from sklearn.ensemble import (
                HistGradientBoostingRegressor
            )

            from sklearn.model_selection import (
                GroupShuffleSplit
            )

            base_dir = Path(__file__).resolve().parent

            labels_path = base_dir / "runtime-data.csv"
            circuits_dir = (
                base_dir.parent
                / "training_circuits"
            )

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

            circuit_features = {}

            for filename, _, _ in training_rows:

                if filename in circuit_features:
                    continue

                compressed_path = (
                    circuits_dir
                    / f"{filename}.zst"
                )

                with open(compressed_path, "rb") as f:
                    dctx = zstd.ZstdDecompressor()

                    with dctx.stream_reader(f) as reader:
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

                    n_qubits * log_ops,
                    n_qubits * two_q_ratio,
                    log_ops * log_threshold,
                    n_qubits * log_threshold,
                    n_2q * log_threshold,

                    n_qubits * n_qubits,
                    log_ops * log_ops,
                    two_q_ratio * log_threshold,
                ]

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
            # Same 2/3 / 1/3 circuit split
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
                "Training circuits: "
                f"{len(set(groups[fit_idx]))}"
            )

            print(
                "Held-out circuits: "
                f"{len(set(groups[test_idx]))}"
            )

            # ---------------------------------------------------------- #
            # Histogram Gradient Boosting
            # ---------------------------------------------------------- #
            model = (
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=500,
                    max_leaf_nodes=31,
                    max_depth=None,
                    min_samples_leaf=10,
                    l2_regularization=0.5,
                    random_state=42,
                )
            )

            model.fit(
                X_fit,
                y_fit,
            )

            test_log_predictions = (
                model.predict(
                    X_test
                )
            )

            actual_seconds = (
                10 ** y_test
            )

            predicted_seconds = (
                10
                ** test_log_predictions
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
                np.mean(
                    held_out_scores
                )
                * 100
            )

            print()
            print(
                "----------------------------------"
            )
            print(
                "HIST GRADIENT BOOSTING "
                "HELD-OUT SCORE: "
                f"{held_out_score:.2f}%"
            )
            print(
                "----------------------------------"
            )
            print()

            self.model = model

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

        predicted_log_seconds = (
            self.model.predict(
                X
            )[0]
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