import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor

from model import CAP_SECONDS, CircuitFeatureParser, flatten_features
from run import find_circuits, read_qasm


HARNESS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = HARNESS_DIR.parent
RUN_SCRIPT = HARNESS_DIR / "run.py"
MAX_TRAINING_ROWS = 1000


def load_labels(labels_path: Path, limit: int) -> list[tuple[str, int, float]]:
    labels_content = labels_path.read_text(encoding="utf-8").strip()
    if not labels_content.startswith("filename,"):
        labels_path = (labels_path.parent / labels_content).resolve()

    labels = []
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        for index, row in enumerate(csv.DictReader(csv_file)):
            if limit and index >= limit:
                break
            if row["status"] == "timeout":
                duration = CAP_SECONDS
            else:
                duration = float(row["duration_s"])
            labels.append((row["filename"], int(row["threshold"]), duration))
    return labels


def build_training_data(circuits_dir: Path, labels_path: Path, limit: int):
    circuit_paths = {
        path.name[:-4] if path.suffix == ".zst" else path.name: path
        for path in find_circuits(circuits_dir)
    }
    labels = load_labels(labels_path, limit)
    if not labels:
        raise ValueError(f"No runtime rows found in {labels_path}")
    if not circuit_paths:
        raise ValueError(f"No QASM files found in {circuits_dir}")

    feature_parser = CircuitFeatureParser(model_features_only=True)
    feature_cache = {}

    for filename, threshold, duration in labels:
        if filename not in circuit_paths:
            raise ValueError(f"No circuit file found for labeled circuit {filename}")
        if filename not in feature_cache:
            parsed = feature_parser.featurize(read_qasm(circuit_paths[filename]))
            feature_cache[filename] = flatten_features(parsed)

    feature_names = tuple(sorted({
        name
        for features in feature_cache.values()
        for name in features
    }))
    model_feature_names = feature_names + ("threshold",)
    samples = []
    targets = []
    for filename, threshold, duration in labels:
        features = feature_cache[filename]
        samples.append([features.get(name, 0.0) for name in feature_names] + [threshold])
        targets.append(math.log10(max(duration, 1e-6)))

    return samples, targets, model_feature_names, list(feature_cache)


def main():
    parser = argparse.ArgumentParser(
        description="Train the upgraded GBT from runtime labels and the Prosanta feature parser, then write a submission."
    )
    parser.add_argument(
        "--circuits",
        type=Path,
        default=PROJECT_DIR / "training_circuits",
        help="Directory containing .qasm and .qasm.zst training circuits.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=HARNESS_DIR / "runtime-data.csv",
        help="Runtime CSV or a file containing a relative path to the runtime CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_TRAINING_ROWS,
        help="Number of initial runtime-data rows to train on (maximum 1000).",
    )
    parser.add_argument(
        "--model-out",
        "--out",
        dest="model_out",
        type=Path,
        default=HARNESS_DIR / "artifacts" / "gbt_model.joblib",
        help="Path for the trained model artifact.",
    )
    parser.add_argument("--submission", type=Path, default=HARNESS_DIR / "submission.csv")
    parser.add_argument("--team", default="Practice Team")
    parser.add_argument("--thresholds", default="16,64,512")
    args = parser.parse_args()
    if not 1 <= args.limit <= MAX_TRAINING_ROWS:
        parser.error("--limit must be between 1 and 1000")

    samples, targets, feature_names, filenames = build_training_data(
        args.circuits, args.labels, args.limit
    )
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(samples, targets)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_transform": "log10_seconds",
            "trained_filenames": filenames,
        },
        args.model_out,
    )
    print(f"Trained on {len(filenames)} circuits and {len(samples)} labeled runs.")
    print(f"Used {len(feature_names) - 1} circuit features plus threshold.")
    print(f"Saved model to {args.model_out}")

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--team", args.team,
            "--circuits", str(args.circuits),
            "--thresholds", args.thresholds,
            "--out", str(args.submission),
        ],
        cwd=HARNESS_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()