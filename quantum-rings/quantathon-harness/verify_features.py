from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd

from features import (
    CircuitFeatureParser,
    flatten_features,
)


BASE_DIR = Path(__file__).resolve().parent
CIRCUITS_DIR = BASE_DIR.parent / "training_circuits"
CSV_PATH = (
    BASE_DIR.parent.parent
    / "prosanta"
    / "circuit_features_flattened.csv"
)


def read_zst(path):
    with path.open("rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as reader:
            return reader.read().decode(
                "utf-8",
                errors="replace",
            )


def main():
    df = pd.read_csv(CSV_PATH)

    parser = CircuitFeatureParser(
        model_features_only=True
    )

    samples = df["filename"].iloc[
        [0, 50, 100, 200, 300, 400, 531]
    ].tolist()

    checked = 0
    differences = 0

    for filename in samples:
        qasm_path = (
            CIRCUITS_DIR
            / f"{filename}.zst"
        )

        qasm = read_zst(qasm_path)

        parsed = flatten_features(
            parser.featurize(qasm)
        )

        expected = (
            df.loc[
                df["filename"] == filename
            ]
            .iloc[0]
        )

        for col in df.columns:
            if col == "filename":
                continue

            actual_value = float(
                parsed.get(col, 0.0)
            )

            expected_value = float(
                expected[col]
            )

            if not np.isclose(
                actual_value,
                expected_value,
                rtol=1e-7,
                atol=1e-7,
            ):
                differences += 1

                if differences <= 20:
                    print(
                        "DIFFERENCE:",
                        filename,
                        col,
                        "parser=",
                        actual_value,
                        "csv=",
                        expected_value,
                    )

        checked += 1
        print(f"checked {filename}")

    print()
    print("circuits checked:", checked)
    print("feature differences:", differences)


if __name__ == "__main__":
    main()
