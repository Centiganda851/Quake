"""
model.py  --  THIS IS THE ONLY FILE YOUR TEAM NEEDS TO EDIT.

Implement the two methods below. The harness (run.py) takes care of finding
circuits, decompressing them, timing you, and writing the submission file.

Contract
--------
  RuntimeModel()                      load your trained model / weights
  .featurize(qasm_text)   -> dict     parse one circuit into features  (once per circuit)
  .predict(features, thr) -> seconds  predict runtime in seconds       (once per (circuit, threshold))

Rules of the challenge (see README):
  * featurize and predict must each run in <= 15 s per circuit.
  * predict returns a single number: the wall-clock seconds you expect the run to take.
  * There is no separate "timeout" flag. If you think a run will hit the 4-hour cap,
    just predict a duration >= the cap (14400 s). Scoring caps it there for you.

The version below is a trivial BASELINE so the harness runs out of the box.
Replace its guts with your real feature parser and model.
"""

import math
import re
from pathlib import Path

CAP_SECONDS = 4 * 60 * 60  # 4-hour timeout cap


def featurize_qasm(qasm_text: str) -> dict:
    n_qubits = 0
    for match in re.finditer(r"q(?:u)?(?:reg|bit)\s+\w+\s*\[\s*(\d+)\s*\]", qasm_text):
        n_qubits += int(match.group(1))

    lines = [line.strip() for line in qasm_text.splitlines()]
    gate_lines = [line for line in lines if line and not line.startswith(("//", "OPENQASM",
                 "include", "qreg", "creg", "qubit", "bit", "gate"))]
    n_ops = len(gate_lines)
    n_2q = len(re.findall(r"\b(cx|cz|cy|ch|swap|iswap|rzz|rxx|ryy|cp|crx|cry|crz|ecr)\b", qasm_text))

    return {"n_qubits": n_qubits, "n_ops": n_ops, "n_2q": n_2q}


class RuntimeModel:
    def __init__(self, artifacts_dir="artifacts"):
        import joblib

        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            artifacts_path = Path(__file__).resolve().parent / artifacts_path
        artifact_path = artifacts_path / "random_forest_model.joblib"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Trained random-forest model not found: {artifact_path}")

        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        if artifact.get("target_transform") != "log10_seconds":
            raise ValueError(f"Unsupported target transform in {artifact_path}")

    # ------------------------------------------------------------------ #
    # 1) FEATURE PARSER  --  .qasm text  ->  feature dict                 #
    # ------------------------------------------------------------------ #
    def featurize(self, qasm_text: str) -> dict:
        return featurize_qasm(qasm_text)

    # ------------------------------------------------------------------ #
    # 2) MODEL  --  (features, threshold)  ->  predicted seconds          #
    # ------------------------------------------------------------------ #
    def predict(self, features: dict, threshold: int) -> float:
        model_features = [features[name] for name in self.feature_names[:3]] + [threshold]
        predicted_log_seconds = self.model.predict([model_features])[0]
        predicted_log_seconds = min(max(float(predicted_log_seconds), -6), math.log10(CAP_SECONDS))
        predicted_seconds = math.pow(10, predicted_log_seconds)
        return float(min(predicted_seconds, CAP_SECONDS))
