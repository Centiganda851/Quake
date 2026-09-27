import math
from pathlib import Path

import joblib
import numpy as np

from features import (
    CircuitFeatureParser,
    flatten_features,
)
from ml_common import (
    CAP_SECONDS,
    feature_dict_to_row,
)


class RuntimeModel:
    def __init__(self, artifacts_dir="artifacts"):
        base_dir = Path(__file__).resolve().parent

        artifacts_path = Path(artifacts_dir)

        if not artifacts_path.is_absolute():
            artifacts_path = (
                base_dir / artifacts_path
            )

        artifact_path = (
            artifacts_path
            / "final_model.joblib"
        )

        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Final model artifact not found: "
                f"{artifact_path}"
            )

        artifact = joblib.load(
            artifact_path
        )

        self.kind = artifact["kind"]

        self.feature_names = artifact[
            "feature_names"
        ]

        if (
            artifact.get("target_transform")
            != "log10_seconds"
        ):
            raise ValueError(
                "Unexpected target transform."
            )

        if self.kind == "single":
            self.model = artifact["model"]

        elif self.kind == "ensemble":
            self.models = artifact["models"]
            self.model_names = artifact[
                "model_names"
            ]
            self.weights = artifact[
                "weights"
            ]

        else:
            raise ValueError(
                f"Unknown artifact kind: "
                f"{self.kind}"
            )

        self.parser = CircuitFeatureParser(
            model_features_only=True
        )

    def featurize(
        self,
        qasm_text: str,
    ) -> dict:
        features = self.parser.featurize(
            qasm_text
        )

        return flatten_features(
            features
        )

    def predict(
        self,
        features: dict,
        threshold: int,
    ) -> float:
        row = feature_dict_to_row(
            features,
            threshold,
            self.feature_names,
        )

        X = np.asarray(
            [row],
            dtype=float,
        )

        if self.kind == "single":
            pred_log = float(
                self.model.predict(X)[0]
            )

        else:
            pred_log = 0.0

            for (
                name,
                weight,
            ) in zip(
                self.model_names,
                self.weights,
            ):
                pred_log += (
                    float(weight)
                    * float(
                        self.models[
                            name
                        ].predict(X)[0]
                    )
                )

        if not math.isfinite(pred_log):
            return CAP_SECONDS

        seconds = math.pow(
            10.0,
            pred_log,
        )

        seconds = max(
            1e-9,
            seconds,
        )

        seconds = min(
            CAP_SECONDS,
            seconds,
        )

        return float(seconds)
