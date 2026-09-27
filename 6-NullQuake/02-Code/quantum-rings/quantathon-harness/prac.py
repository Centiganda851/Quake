import argparse
import subprocess
import sys
from pathlib import Path


HARNESS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = HARNESS_DIR.parent
RUN_SCRIPT = HARNESS_DIR / "run.py"
SCORE_SCRIPT = HARNESS_DIR / "score.py"
CIRCUITS_DIR = PROJECT_DIR / "holdout-circuits"
LABELS_FILE = PROJECT_DIR / "runtime-data.csv"
SUBMISSION_FILE = HARNESS_DIR / "submission.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Run the prediction harness and score its output on training data."
    )
    parser.add_argument(
        "--team",
        default="NullQuake",
        help="Team name written into the generated submission (default: NullQuake).",
    )
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--team",
            args.team,
            "--circuits",
            str(CIRCUITS_DIR),
            "--out",
            str(SUBMISSION_FILE),
        ],
        cwd=HARNESS_DIR,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(SCORE_SCRIPT),
            "--pred",
            str(SUBMISSION_FILE),
            "--labels",
            str(LABELS_FILE),
        ],
        cwd=HARNESS_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
