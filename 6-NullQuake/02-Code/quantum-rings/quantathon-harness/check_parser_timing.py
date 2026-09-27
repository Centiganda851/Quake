import time
from pathlib import Path

import numpy as np
import zstandard as zstd

from features import CircuitFeatureParser


BASE_DIR = Path(__file__).resolve().parent
CIRCUITS_DIR = BASE_DIR.parent / "training_circuits"


def read_zst(path):
    with path.open("rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as reader:
            return reader.read().decode(
                "utf-8",
                errors="replace",
            )


def main():
    parser = CircuitFeatureParser(
        model_features_only=True
    )

    paths = sorted(
        CIRCUITS_DIR.glob(
            "*.qasm.zst"
        )
    )

    times = []

    for i, path in enumerate(paths, 1):
        qasm = read_zst(path)

        start = time.perf_counter()
        parser.featurize(qasm)
        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(elapsed)

        if elapsed > 15:
            print(
                "OVER 15 SECONDS:",
                path.name,
                elapsed,
            )

        if i % 25 == 0 or i == len(paths):
            print(
                f"{i}/{len(paths)}"
            )

    times = np.asarray(
        times,
        dtype=float,
    )

    print()
    print(
        "mean:",
        times.mean(),
    )
    print(
        "p95:",
        np.percentile(
            times,
            95,
        ),
    )
    print(
        "max:",
        times.max(),
    )
    print(
        "over 15 sec:",
        int(
            np.sum(
                times > 15
            )
        ),
    )


if __name__ == "__main__":
    main()
