import argparse
import csv
import math
from pathlib import Path

import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import CAP_SECONDS, CircuitFeatureParser, flatten_features
from run import find_circuits, read_qasm


HARNESS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = HARNESS_DIR.parent
MAX_TRAINING_ROWS = 1000


def resolve_labels_path(path: Path) -> Path:
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("filename,"):
        return path
    target = (path.parent / content).resolve()
    if target.is_file():
        return target
    raise ValueError(f"Expected a runtime CSV or path pointer in {path}")


def load_training_rows(labels_path: Path, limit: int):
    labels_path = resolve_labels_path(labels_path)
    rows = []
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        for index, row in enumerate(csv.DictReader(csv_file)):
            if limit and index >= limit:
                break
            duration = (
                CAP_SECONDS
                if row["status"].strip().lower() == "timeout"
                else float(row["duration_s"])
            )
            rows.append((row["filename"].strip(), int(row["threshold"]), duration))
    if not rows:
        raise ValueError(f"No runtime rows found in {labels_path}")
    return rows


def build_training_data(circuits_dir: Path, labels_path: Path, limit: int):
    circuit_paths = {
        path.name[:-4] if path.suffix == ".zst" else path.name: path
        for path in find_circuits(circuits_dir)
    }
    labels = load_training_rows(labels_path, limit)
    parser = CircuitFeatureParser(model_features_only=True)
    feature_cache = {}

    for filename, _, _ in labels:
        if filename not in circuit_paths:
            raise ValueError(f"No circuit file found for labeled circuit {filename}")
        if filename not in feature_cache:
            parsed = parser.featurize(read_qasm(circuit_paths[filename]))
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

    return samples, targets, model_feature_names, sorted(feature_cache)


def main():
    parser = argparse.ArgumentParser(
        description="Train KNN using all integrated runtime thresholds and upgraded QASM features."
    )
    parser.add_argument(
        "--circuits",
        type=Path,
        default=PROJECT_DIR / "training_circuits",
        help="Directory containing training .qasm and .qasm.zst files.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=HARNESS_DIR / "runtime-data.csv",
        help="Runtime CSV or a file containing a relative path to one.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_TRAINING_ROWS,
        help="Number of initial runtime-data rows to train on (maximum 1000).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HARNESS_DIR / "artifacts" / "knn_model.joblib",
        help="Output KNN model artifact.",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= MAX_TRAINING_ROWS:
        parser.error("--limit must be between 1 and 1000")

    samples, targets, feature_names, filenames = build_training_data(
        args.circuits, args.labels, args.limit
    )
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(
            n_neighbors=min(5, len(samples)),
            weights="distance",
            p=2,
        ),
    )
    model.fit(samples, targets)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_transform": "log10_seconds",
            "trained_filenames": filenames,
        },
        args.out,
    )
    print(f"Trained on {len(filenames)} circuits and {len(samples)} runtime rows.")
    print(f"Used {len(feature_names) - 1} circuit features plus threshold.")
    print(f"Saved KNN model to {args.out}")


if __name__ == "__main__":
    main()
