from collections import defaultdict
from itertools import combinations
import re


LARGE_QASM_BYTES = 40_000_000
SAMPLE_CHUNK_BYTES = 1_000_000


_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_QREG2_RE = re.compile(
    rf"^qreg\s+({_IDENT})\s*\[\s*(\d+)\s*\]\s*$", re.IGNORECASE
)
_QUBIT3_ARRAY_RE = re.compile(
    rf"^qubit\s*\[\s*(\d+)\s*\]\s*({_IDENT})\s*$", re.IGNORECASE
)
_QUBIT3_ALT_RE = re.compile(
    rf"^qubit\s+({_IDENT})\s*\[\s*(\d+)\s*\]\s*$", re.IGNORECASE
)
_QUBIT3_SCALAR_RE = re.compile(rf"^qubit\s+({_IDENT})\s*$", re.IGNORECASE)
_CREG2_RE = re.compile(
    rf"^creg\s+({_IDENT})\s*\[\s*(\d+)\s*\]\s*$", re.IGNORECASE
)
_BIT3_ARRAY_RE = re.compile(
    rf"^bit\s*\[\s*(\d+)\s*\]\s*({_IDENT})\s*$", re.IGNORECASE
)
_BIT3_SCALAR_RE = re.compile(rf"^bit\s+({_IDENT})\s*$", re.IGNORECASE)
_OP_RE = re.compile(
    rf"^({_IDENT})\s*(\([^;]*\))?\s+(.+)$", re.IGNORECASE
)
_REF_RE = re.compile(
    rf"^({_IDENT})(?:\s*\[\s*(\d+)(?:\s*:\s*(\d+))?\s*\])?$"
)
_HARDWARE_QUBIT_RE = re.compile(r"^\$(\d+)$")
_LEADING_IF_RE = re.compile(r"^if\s*\([^)]*\)\s*", re.IGNORECASE)
_DECLARATION_LINE_RE = re.compile(
    r"(?m)^(?:qubit|qreg|bit|creg)\b", re.IGNORECASE
)
_MCPHASE_VARIANT_RE = re.compile(r"^mcphase(?:_\d+)?$")
_GENERATED_CIRCUIT_GATE_RE = re.compile(
    r"^_c?circuit_\d+(?:_\d+)?(?:_dg)?$"
)

# These fields are useful for diagnostics but are not recommended as initial
# simulator-runtime inputs. They can introduce format, classical-register, or
# parser-strategy signals that may not generalize to hidden circuits.
MODEL_EXCLUDED_FEATURES = frozenset({
    "qasm_version",
    "n_clbits",
    "measurement_count",
    "barrier_count",
    "delay_count",
    "sampled",
    "sample_fraction",
    "definitions_skipped",
    "source_fraction",
})


_SKIP_PREFIXES = (
    "openqasm",
    "include",
    "input",
    "output",
    "const",
    "let",
    "int",
    "uint",
    "float",
    "angle",
    "bool",
    "duration",
    "stretch",
    "extern",
    "pragma",
)

_CLIFFORD_GATES = {
    "id", "i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg",
    "cx", "cnot", "cy", "cz", "ch", "swap", "ecr",
}
_DIAGONAL_GATES = {
    "z", "s", "sdg", "t", "tdg", "p", "u1", "rz", "cz", "cp",
    "cu1", "rzz", "ccz",
}
_ROTATION_GATES = {
    "p", "u", "u1", "u2", "u3", "rx", "ry", "rz", "rxx", "ryy",
    "rzz", "rzx", "cp", "crx", "cry", "crz", "cu", "cu1", "cu2",
    "cu3", "xx_plus_yy", "xx_minus_yy",
}


def _strip_comments(line, in_block_comment):
    if not in_block_comment and "/" not in line:
        return line, False
    pieces = []
    pos = 0
    size = len(line)
    while pos < size:
        if in_block_comment:
            end = line.find("*/", pos)
            if end < 0:
                return "".join(pieces), True
            pos = end + 2
            in_block_comment = False
            continue

        line_comment = line.find("//", pos)
        block_comment = line.find("/*", pos)
        if line_comment >= 0 and (block_comment < 0 or line_comment < block_comment):
            pieces.append(line[pos:line_comment])
            break
        if block_comment < 0:
            pieces.append(line[pos:])
            break
        pieces.append(line[pos:block_comment])
        pos = block_comment + 2
        in_block_comment = True
    return "".join(pieces), in_block_comment


def _iter_statements(qasm_text):
    pending = ""
    in_block_comment = False
    definition_depth = 0
    waiting_for_definition_body = False

    # splitlines() would allocate a second, very large list for a 200 MB file
    start = 0
    text_size = len(qasm_text)
    while start < text_size:
        end = qasm_text.find("\n", start)
        if end < 0:
            end = text_size
        raw_line = qasm_text[start:end]
        start = end + 1

        line, in_block_comment = _strip_comments(raw_line, in_block_comment)
        stripped = line.strip()
        if not stripped:
            continue

        if definition_depth:
            definition_depth += stripped.count("{") - stripped.count("}")
            if definition_depth <= 0:
                definition_depth = 0
            continue

        if waiting_for_definition_body:
            if "{" in stripped:
                definition_depth = stripped.count("{") - stripped.count("}")
                waiting_for_definition_body = False
            continue

        # Definitions are rare. Avoid split/lower allocation on every gate.
        lowered_start = stripped[:8].lower()
        first_word = lowered_start.split(None, 1)[0]
        if first_word in {"gate", "def", "defcal"}:
            pending = ""
            if "{" in stripped:
                definition_depth = stripped.count("{") - stripped.count("}")
            elif first_word != "gate" or not stripped.endswith(";"):
                waiting_for_definition_body = True
            continue
        if first_word == "opaque":
            pending = ""
            continue

        # The challenge data normally has exactly one statement on each line.
        # This fast path matters for multi-million-gate circuits.
        if (
            not pending
            and "{" not in stripped
            and "}" not in stripped
            and stripped.endswith(";")
            and stripped.count(";") == 1
        ):
            yield stripped[:-1].strip()
            continue

        # Braces belonging to ordinary QASM 3 control flow are separators.
        line = line.replace("{", " ").replace("}", " ")
        pending = f"{pending} {line}" if pending else line
        while True:
            semi = pending.find(";")
            if semi < 0:
                break
            statement = pending[:semi].strip()
            pending = pending[semi + 1 :]
            if statement:
                yield statement

    if pending.strip():
        # Helpful for permissive parsing of a final statement with no semicolon.
        yield pending.strip()


def _connected_components(adjacency):
    remaining = set(adjacency)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            unseen = adjacency[node] & remaining
            if unseen:
                remaining.difference_update(unseen)
                stack.extend(unseen)
    return count


def _min_degree_treewidth(adjacency):
    """Return a min-degree upper bound; exact treewidth is NP-hard."""
    graph = {node: set(neighbors) for node, neighbors in adjacency.items()}
    width = 0
    while graph:
        node = min(graph, key=lambda item: len(graph[item]))
        neighbors = graph[node]
        width = max(width, len(neighbors))
        for left, right in combinations(neighbors, 2):
            graph[left].add(right)
            graph[right].add(left)
        for neighbor in neighbors:
            graph[neighbor].discard(node)
        del graph[node]
    return width


def _normalize_gate_name(name):
    """Collapse automatically numbered custom gates into stable categories."""
    if _MCPHASE_VARIANT_RE.fullmatch(name):
        return "mcphase"
    if _GENERATED_CIRCUIT_GATE_RE.fullmatch(name):
        return "custom_circuit"
    return name


class CircuitFeatureParser:
    def __init__(self, model_features_only=True):
        self.model_features_only = model_features_only

    def _select_features(self, features):
        """Return the model input vector or the complete diagnostic vector."""
        if not self.model_features_only:
            return features

        selected = {
            name: value
            for name, value in features.items()
            if name not in MODEL_EXCLUDED_FEATURES
        }
        selected["inactive_qubits"] = max(
            0, selected["n_qubits"] - selected["active_qubits"]
        )
        return selected

    @staticmethod
    def _resolve_operand(token, registers):
        token = token.strip()
        match = _HARDWARE_QUBIT_RE.fullmatch(token)
        if match:
            return [int(match.group(1))]

        match = _REF_RE.fullmatch(token)
        if not match:
            return []
        name, first, last = match.groups()
        if name not in registers:
            return []
        offset, size = registers[name]
        if first is None:
            return list(range(offset, offset + size))
        first = int(first)
        if last is None:
            return [offset + first] if first < size else []
        last = int(last)
        step = 1 if last >= first else -1
        return [
            offset + index
            for index in range(first, last + step, step)
            if 0 <= index < size
        ]

    # ------------------------------------------------------------------ 
    # FEATURE PARSER: QASM text -> feature dictionary              
    # ------------------------------------------------------------------
    def featurize(self, qasm_text: str) -> dict:
        if len(qasm_text) > LARGE_QASM_BYTES:
            return self._select_features(self._featurize_large(qasm_text))

        registers = {}
        n_qubits = 0
        n_clbits = 0
        last_layer = []

        gate_counts = defaultdict(int)
        interaction_weights = defaultdict(int)
        per_qubit_ops = []
        operand_cache = {}

        n_ops = 0
        n_1q = 0
        n_2q = 0
        n_multiq = 0
        max_gate_arity = 0
        measurement_count = 0
        reset_count = 0
        barrier_count = 0
        delay_count = 0
        parameterized_gate_count = 0
        diagonal_gate_count = 0
        rotation_gate_count = 0
        clifford_gate_count = 0
        t_gate_count = 0
        s_gate_count = 0

        def ensure_qubits(count):
            nonlocal n_qubits
            if count > n_qubits:
                growth = count - n_qubits
                last_layer.extend([0] * growth)
                per_qubit_ops.extend([0] * growth)
                n_qubits = count

        def add_register(name, size):
            nonlocal n_qubits
            if name in registers:
                return
            registers[name] = (n_qubits, size)
            ensure_qubits(n_qubits + size)

        def resolve_operand(token):
            token = token.strip()
            cached = operand_cache.get(token)
            if cached is not None:
                return cached
            resolved = self._resolve_operand(token, registers)
            operand_cache[token] = resolved
            return resolved

        def record_operation(gate_name, qubits, has_parameters):
            nonlocal n_ops, n_1q, n_2q, n_multiq, max_gate_arity
            nonlocal parameterized_gate_count, diagonal_gate_count
            nonlocal rotation_gate_count, clifford_gate_count
            nonlocal t_gate_count, s_gate_count

            if not qubits:
                return
            if len(qubits) == 2:
                if qubits[0] == qubits[1]:
                    qubits = qubits[:1]
                elif qubits[0] > qubits[1]:
                    qubits = [qubits[1], qubits[0]]
            elif len(qubits) > 2:
                qubits = sorted(set(qubits))
            ensure_qubits(max(qubits) + 1)
            arity = len(qubits)
            n_ops += 1
            gate_counts[gate_name] += 1
            max_gate_arity = max(max_gate_arity, arity)
            if arity == 1:
                n_1q += 1
            elif arity == 2:
                n_2q += 1
            else:
                n_multiq += 1

            if has_parameters:
                parameterized_gate_count += 1
            if gate_name in _DIAGONAL_GATES:
                diagonal_gate_count += 1
            if gate_name in _ROTATION_GATES:
                rotation_gate_count += 1
            if gate_name in _CLIFFORD_GATES:
                clifford_gate_count += 1
            if gate_name in {"t", "tdg"}:
                t_gate_count += 1
            if gate_name in {"s", "sdg"}:
                s_gate_count += 1

            if arity == 1:
                index = qubits[0]
                layer = last_layer[index] + 1
                last_layer[index] = layer
                per_qubit_ops[index] += 1
            elif arity == 2:
                left, right = qubits
                layer = 1 + max(last_layer[left], last_layer[right])
                last_layer[left] = layer
                last_layer[right] = layer
                per_qubit_ops[left] += 1
                per_qubit_ops[right] += 1
                interaction_weights[(left, right)] += 1
            else:
                layer = 1 + max(last_layer[index] for index in qubits)
                for index in qubits:
                    last_layer[index] = layer
                    per_qubit_ops[index] += 1
                for left, right in combinations(qubits, 2):
                    interaction_weights[(left, right)] += 1

        for statement in _iter_statements(qasm_text):
            lowered = statement.lower().strip()

            if lowered.startswith("qreg "):
                match = _QREG2_RE.fullmatch(statement)
                if match:
                    add_register(match.group(1), int(match.group(2)))
                    continue
            elif lowered.startswith("qubit"):
                match = _QUBIT3_ARRAY_RE.fullmatch(statement)
                if match:
                    add_register(match.group(2), int(match.group(1)))
                    continue
                match = _QUBIT3_ALT_RE.fullmatch(statement)
                if match:
                    add_register(match.group(1), int(match.group(2)))
                    continue
                match = _QUBIT3_SCALAR_RE.fullmatch(statement)
                if match:
                    add_register(match.group(1), 1)
                    continue
            elif lowered.startswith("creg "):
                match = _CREG2_RE.fullmatch(statement)
                if match:
                    n_clbits += int(match.group(2))
                    continue
            elif lowered.startswith("bit"):
                match = _BIT3_ARRAY_RE.fullmatch(statement)
                if match:
                    n_clbits += int(match.group(1))
                    continue
                match = _BIT3_SCALAR_RE.fullmatch(statement)
                if match:
                    n_clbits += 1
                    continue

            if lowered.startswith(_SKIP_PREFIXES):
                continue

            # QASM 3 assignment and QASM 2 arrow forms both contain "measure".
            if "measure" in lowered:
                measurement_count += 1
                continue
            if lowered.startswith("barrier"):
                barrier_count += 1
                operand_text = statement[len("barrier") :]
                groups = [
                    resolve_operand(part)
                    for part in operand_text.split(",")
                ]
                qubits = sorted({q for group in groups for q in group})
                if qubits:
                    layer = max(last_layer[q] for q in qubits)
                    for q in qubits:
                        last_layer[q] = layer
                continue
            if lowered.startswith("delay"):
                delay_count += 1
                continue

            statement = _LEADING_IF_RE.sub("", statement).strip()
            # In QASM 3 modifiers such as "ctrl @ inv @ x", the actual gate
            # name and operands occur after the last @.
            if "@" in statement:
                statement = statement.rsplit("@", 1)[1].strip()

            parameter_start = statement.find("(")
            first_space = statement.find(" ")
            if parameter_start >= 0 and (
                first_space < 0 or parameter_start < first_space
            ):
                parameter_end = statement.rfind(")")
                if parameter_end < parameter_start:
                    continue
                gate_name = statement[:parameter_start].strip().lower()
                has_parameters = True
                operand_text = statement[parameter_end + 1 :].strip()
            elif first_space > 0:
                gate_name = statement[:first_space].lower()
                has_parameters = False
                operand_text = statement[first_space + 1 :].strip()
            else:
                match = _OP_RE.fullmatch(statement)
                if not match:
                    continue
                gate_name = match.group(1).lower()
                has_parameters = bool(match.group(2))
                operand_text = match.group(3).strip()

            if gate_name == "reset":
                reset_count += 1
            if "->" in operand_text:
                operand_text = operand_text.split("->", 1)[0]

            groups = [
                resolve_operand(part)
                for part in operand_text.split(",")
            ]
            groups = [group for group in groups if group]
            if not groups:
                continue

            gate_name = _normalize_gate_name(gate_name)

            # OpenQASM register broadcasting applies the gate elementwise.
            applications = max(len(group) for group in groups)
            compatible = all(len(group) in (1, applications) for group in groups)
            if compatible:
                for index in range(applications):
                    qubits = [
                        group[index] if len(group) > 1 else group[0]
                        for group in groups
                    ]
                    record_operation(gate_name, qubits, has_parameters)
            else:
                # Preserve signal from unusual slices rather than failing the
                # whole circuit.
                record_operation(
                    gate_name,
                    [q for group in groups for q in group],
                    has_parameters,
                )

        depth = max(last_layer, default=0)
        active_qubits = sum(count > 0 for count in per_qubit_ops)
        max_qubit_load = max(per_qubit_ops, default=0)
        avg_qubit_load = sum(per_qubit_ops) / n_qubits if n_qubits else 0.0

        adjacency = {index: set() for index in range(n_qubits)}
        weighted_degree = [0] * n_qubits
        weighted_distance = 0
        pair_interactions = 0
        max_2q_dist = 0
        cut_deltas = [0] * (n_qubits + 1)
        for (left, right), weight in interaction_weights.items():
            adjacency[left].add(right)
            adjacency[right].add(left)
            weighted_degree[left] += weight
            weighted_degree[right] += weight
            distance = right - left
            weighted_distance += distance * weight
            pair_interactions += weight
            max_2q_dist = max(max_2q_dist, distance)
            cut_deltas[left] += weight
            cut_deltas[right] -= weight

        running_cut = 0
        max_cutwidth = 0
        for index in range(max(0, n_qubits - 1)):
            running_cut += cut_deltas[index]
            max_cutwidth = max(max_cutwidth, running_cut)

        unique_interactions = len(interaction_weights)
        possible_edges = n_qubits * (n_qubits - 1) / 2
        graph_density = (
            unique_interactions / possible_edges if possible_edges else 0.0
        )
        component_count = _connected_components(adjacency) if adjacency else 0
        treewidth = _min_degree_treewidth(adjacency) if adjacency else 0

        features = {
            "qasm_version": (
                3 if re.search(r"OPENQASM\s+3", qasm_text[:200], re.I) else 2
            ),
            "n_qubits": n_qubits,
            "n_clbits": n_clbits,
            "n_ops": n_ops,
            "n_1q": n_1q,
            "n_2q": n_2q,
            "n_multiq": n_multiq,
            "depth": depth,
            "max_gate_arity": max_gate_arity,
            "active_qubits": active_qubits,
            "avg_qubit_load": avg_qubit_load,
            "max_qubit_load": max_qubit_load,
            "two_qubit_gate_density": n_2q / n_ops if n_ops else 0.0,
            "entangling_gate_density": (
                (n_2q + n_multiq) / n_ops if n_ops else 0.0
            ),
            "parallelism": n_ops / depth if depth else 0.0,
            "unique_interactions": unique_interactions,
            "graph_density": graph_density,
            "connected_components": component_count,
            "treewidth": treewidth,
            "avg_2q_dist": (
                weighted_distance / pair_interactions if pair_interactions else 0.0
            ),
            "max_2q_dist": max_2q_dist,
            "max_cutwidth": max_cutwidth,
            "max_weighted_degree": max(weighted_degree, default=0),
            "measurement_count": measurement_count,
            "reset_count": reset_count,
            "barrier_count": barrier_count,
            "delay_count": delay_count,
            "parameterized_gate_count": parameterized_gate_count,
            "diagonal_gate_count": diagonal_gate_count,
            "rotation_gate_count": rotation_gate_count,
            "clifford_gate_count": clifford_gate_count,
            "t_gate_count": t_gate_count,
            "s_gate_count": s_gate_count,
            "gate_counts": dict(gate_counts),
            "sampled": False,
            "sample_fraction": 1.0,
            "definitions_skipped": False,
            "source_fraction": 1.0,
        }
        return self._select_features(features)

    def _featurize_large(self, qasm_text):
        """Sample huge files so parsing remains safely below the 15 s cap"""
        size = len(qasm_text)
        declaration = _DECLARATION_LINE_RE.search(qasm_text)
        if declaration and declaration.start() > SAMPLE_CHUNK_BYTES:
            version = 3 if re.search(r"OPENQASM\s+3", qasm_text[:200], re.I) else 2
            executable_text = (
                f"OPENQASM {version}.0;\n" + qasm_text[declaration.start() :]
            )
            if len(executable_text) <= LARGE_QASM_BYTES:
                features = self.featurize(executable_text)
                features["definitions_skipped"] = True
                features["source_fraction"] = len(executable_text) / size
                return features

        def complete_chunk(start, length):
            if start > 0:
                start = qasm_text.find("\n", start)
                if start < 0:
                    return ""
                start += 1
            end = min(size, start + length)
            if end < size:
                line_end = qasm_text.find("\n", end)
                end = size if line_end < 0 else line_end + 1
            return qasm_text[start:end]

        starts = (0, size // 3, (2 * size) // 3, max(0, size - SAMPLE_CHUNK_BYTES))
        chunks = [complete_chunk(start, SAMPLE_CHUNK_BYTES) for start in starts]
        sample_text = "\n".join(chunk for chunk in chunks if chunk)
        features = self.featurize(sample_text)

        full_statements = qasm_text.count(";")
        sample_statements = max(sample_text.count(";"), 1)
        sample_ops = features["n_ops"]
        # Header/declaration statements are captured in the first window and
        # terminal measurements are captured in the last one.
        sampled_non_ops = max(0, sample_statements - sample_ops)
        estimated_ops = max(0, full_statements - sampled_non_ops)
        scale = estimated_ops / sample_ops if sample_ops else 1.0

        integer_counts = (
            "n_1q", "n_2q", "n_multiq", "depth",
            "max_qubit_load", "max_cutwidth", "max_weighted_degree",
            "parameterized_gate_count", "diagonal_gate_count",
            "rotation_gate_count", "clifford_gate_count", "t_gate_count",
            "s_gate_count",
        )
        for key in integer_counts:
            features[key] = int(round(features[key] * scale))
        features["n_ops"] = estimated_ops
        features["avg_qubit_load"] *= scale
        features["gate_counts"] = {
            gate: int(round(count * scale))
            for gate, count in features["gate_counts"].items()
        }

        n_ops = features["n_ops"]
        features["two_qubit_gate_density"] = (
            features["n_2q"] / n_ops if n_ops else 0.0
        )
        features["entangling_gate_density"] = (
            (features["n_2q"] + features["n_multiq"]) / n_ops
            if n_ops else 0.0
        )
        features["parallelism"] = (
            n_ops / features["depth"] if features["depth"] else 0.0
        )
        features["sampled"] = True
        features["sample_fraction"] = len(sample_text) / size
        features["definitions_skipped"] = False
        features["source_fraction"] = 1.0
        return features

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


def flatten_features(features: dict) -> dict[str, float]:
    flattened = {
        name: float(value)
        for name, value in features.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }
    flattened.update(
        {
            f"gate_count_{name}": float(count)
            for name, count in features.get("gate_counts", {}).items()
        }
    )
    return flattened


class RuntimeModel:
    def __init__(self, artifacts_dir="artifacts"):
        import joblib

        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            artifacts_path = Path(__file__).resolve().parent / artifacts_path
        artifact_path = artifacts_path / "gbt_model.joblib"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Trained GBT model not found: {artifact_path}")

        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        if artifact.get("target_transform") != "log10_seconds":
            raise ValueError(f"Unsupported target transform in {artifact_path}")
        self.feature_parser = CircuitFeatureParser(model_features_only=True)

    # ------------------------------------------------------------------ #
    # 1) FEATURE PARSER  --  .qasm text  ->  feature dict                 #
    # ------------------------------------------------------------------ #
    def featurize(self, qasm_text: str) -> dict:
        return flatten_features(self.feature_parser.featurize(qasm_text))

    # ------------------------------------------------------------------ #
    # 2) MODEL  --  (features, threshold)  ->  predicted seconds          #
    # ------------------------------------------------------------------ #
    def predict(self, features: dict, threshold: int) -> float:
        model_features = [
            float(threshold) if name == "threshold" else features.get(name, 0.0)
            for name in self.feature_names
        ]
        predicted_log_seconds = self.model.predict([model_features])[0]
        predicted_seconds = math.pow(10, predicted_log_seconds)
        return float(min(predicted_seconds, CAP_SECONDS))
