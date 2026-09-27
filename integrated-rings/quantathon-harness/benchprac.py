import argparse
import subprocess
import sys
from pathlib import Path


HARNESS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = HARNESS_DIR.parent
TRAIN_SCRIPT = HARNESS_DIR / "KNN_gen.py"
BENCHMARK_SCRIPT = HARNESS_DIR / "benchmark.py"
LABELS_FILE = HARNESS_DIR / "runtime-data.csv"
CIRCUITS_DIR = PROJECT_DIR / "training_circuits"
MODEL_ARTIFACT = "knn_model.joblib"
SUBMISSION_FILE = HARNESS_DIR / "knn_submission.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Train the integrated KNN model and report overall and per-threshold accuracy."
    )
    parser.add_argument("--team", default="KNN Practice")
    parser.add_argument("--thresholds", default="16,64,512")
    parser.add_argument("--circuits", type=Path, default=CIRCUITS_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_FILE)
    parser.add_argument("--submission", type=Path, default=SUBMISSION_FILE)
    args = parser.parse_args()

    subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=HARNESS_DIR,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--team", args.team,
            "--circuits", str(args.circuits),
            "--thresholds", args.thresholds,
            "--model-artifact", MODEL_ARTIFACT,
            "--labels", str(args.labels),
            "--out", str(args.submission),
        ],
        cwd=HARNESS_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
