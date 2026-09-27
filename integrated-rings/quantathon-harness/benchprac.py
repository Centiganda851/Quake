import argparse
import joblib
import subprocess
import sys
from pathlib import Path


HARNESS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = HARNESS_DIR.parent
BENCHMARK_SCRIPT = HARNESS_DIR / "benchmark.py"
LABELS_FILE = HARNESS_DIR / "runtime-data.csv"
CIRCUITS_DIR = PROJECT_DIR / "training_circuits"
MODEL_ARTIFACT = "knn_model.joblib"
MODEL_PATH = HARNESS_DIR / "artifacts" / MODEL_ARTIFACT
SUBMISSION_FILE = HARNESS_DIR / "knn_submission.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the existing integrated KNN model and report overall and per-threshold accuracy."
    )
    parser.add_argument("--team", default="KNN Practice")
    parser.add_argument("--thresholds", default="16,64,512")
    parser.add_argument("--circuits", type=Path, default=CIRCUITS_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_FILE)
    parser.add_argument("--submission", type=Path, default=SUBMISSION_FILE)
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        parser.error(f"KNN model artifact not found: {MODEL_PATH}. Train it with KNN_gen.py first.")

    artifact = joblib.load(MODEL_PATH)
    threshold_scores = artifact.get("cv_threshold_scores")
    if artifact.get("cv_folds") != 5 or not threshold_scores:
        parser.error("KNN artifact has no per-threshold 5-fold scores. Run KNN_gen.py to refresh it.")

    print("Five-fold grouped cross-validation accuracy (out-of-fold):", flush=True)
    print(f"  Overall: {artifact['cv_overall_accuracy']:.2%}", flush=True)
    for threshold in (16, 64, 512):
        score = threshold_scores.get(str(threshold))
        if score is not None:
            print(f"  Threshold {threshold}: {score:.2%}", flush=True)

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
