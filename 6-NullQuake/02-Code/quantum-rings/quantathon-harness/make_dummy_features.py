import hashlib
import pandas as pd


RUNTIME_FILE = "runtime-data.csv"
OUTPUT_FILE = "dummy_features.csv"


def stable_number(text: str, minimum: int, maximum: int, salt: str = "") -> int:
    """
    Convert a string into a deterministic pseudo-random integer.

    Same filename always produces the same value.
    This is ONLY for generating dummy test data.
    """
    value = hashlib.sha256((salt + text).encode()).hexdigest()
    number = int(value[:12], 16)

    return minimum + (number % (maximum - minimum + 1))


def main():
    runtime_df = pd.read_csv(RUNTIME_FILE)

    # We need exactly one feature row per unique circuit.
    filenames = sorted(runtime_df["filename"].unique())

    rows = []

    for filename in filenames:
        # THESE ARE FAKE FEATURES.
        # They only exist so we can test the modeling pipeline.
        n_qubits = stable_number(filename, 2, 260, "qubits")
        n_ops = stable_number(filename, 20, 50000, "ops")
        n_2q = stable_number(filename, 0, max(1, n_ops // 2), "2q")
        n_1q = max(0, n_ops - n_2q)

        depth = stable_number(filename, 5, max(5, n_ops), "depth")

        rows.append(
            {
                "filename": filename,
                "n_qubits": n_qubits,
                "n_ops": n_ops,
                "n_1q": n_1q,
                "n_2q": n_2q,
                "two_q_ratio": n_2q / max(n_ops, 1),
                "approx_depth": depth,
            }
        )

    features_df = pd.DataFrame(rows)

    features_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created {OUTPUT_FILE}")
    print(f"Circuits: {len(features_df)}")
    print()
    print(features_df.head())


if __name__ == "__main__":
    main()