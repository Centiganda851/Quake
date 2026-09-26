import argparse
import csv
import math
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor

from model import CAP_SECONDS, RuntimeModel
from run import find_circuits, read_qasm


FEATURE_NAMES = ("n_qubits", "n_ops", "n_2q", "threshold")
HARNESS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = HARNESS_DIR.parent


def load_labels(labels_path: Path, limit: int) -> list[tuple[str, int, float]]:
    labels = []
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        for index, row in enumerate(csv.DictReader(csv_file)):
            if index >= limit:
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

    feature_parser = RuntimeModel()
    feature_cache = {}
    samples = []
    targets = []

    for filename, threshold, duration in labels:
        if filename not in circuit_paths:
            raise ValueError(f"No circuit file found for labeled circuit {filename}")
        if filename not in feature_cache:
            feature_cache[filename] = feature_parser.featurize(
                read_qasm(circuit_paths[filename])
            )
        features = feature_cache[filename]
        samples.append([features[name] for name in FEATURE_NAMES[:3]] + [threshold])
        targets.append(math.log10(max(duration, 1e-6)))

    return samples, targets, list(feature_cache)


def main():
    parser = argparse.ArgumentParser(
        description="Train a gradient-boosting runtime model from the first N runtime-data rows."
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
        default=PROJECT_DIR / "runtime-data.csv",
        help="CSV containing measured runtimes keyed by filename and threshold.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Number of runtime-data rows to train on.")
    parser.add_argument(
        "--out",
        type=Path,
        default=HARNESS_DIR / "artifacts" / "gbt_model.joblib",
        help="Path for the trained model artifact.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    samples, targets, filenames = build_training_data(args.circuits, args.labels, args.limit)
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(samples, targets)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "target_transform": "log10_seconds",
            "trained_filenames": filenames,
        },
        args.out,
    )
    print(f"Trained on {len(filenames)} circuits and {len(samples)} labeled runs.")
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()