"""Check RTL results written by a testbench against the NumPy golden model.

    verify_results.py matmul  RESULTS_FILE
    verify_results.py layer   RESULTS_FILE
    verify_results.py network RESULTS_FILE

The testbenches write these files with +resultsfile=<path>.

matmul (tb/matrix_mult_tb.sv):        layer (tb/nn_layer_tb.sv):

    dims M K N                            dims INPUT_SIZE OUTPUT_SIZE RELU
    case <name>                           case <name>
    A <M*K integers, row-major>           X <INPUT_SIZE integers>
    B <K*N integers, row-major>           W <INPUT_SIZE*OUTPUT_SIZE, row-major>
    C <M*N integers, from the DUT>        B <OUTPUT_SIZE integers>
    ...                                   Y <OUTPUT_SIZE integers, from the DUT>
    end <number of cases>                 ...
                                          end <number of cases>

network (tb/nn_accelerator_tb.sv):

    dims INPUT_SIZE HIDDEN_SIZE OUTPUT_SIZE SHIFT
    case <name>
    X <INPUT_SIZE>   W1 <INPUT_SIZE*HIDDEN_SIZE>   B1 <HIDDEN_SIZE>
    W2 <HIDDEN_SIZE*OUTPUT_SIZE>   B2 <OUTPUT_SIZE>
    Y <OUTPUT_SIZE logits, from the DUT>
    ...
    end <number of cases>

Every value produced by the DUT is recomputed with NumPy and compared. Exits 0
if everything matches, 1 on any mismatch or malformed/truncated file, and 2 on
bad usage.
"""

import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed: run `make venv` to create .venv with numpy")

from golden_model import layer_affine, layer_relu, matmul_int8, network_2layer

MAX_REPORTED = 10

# kind -> (number of dims on the 'dims' line, data tags per case)
RECORD_KINDS = {
    "matmul": (3, ("A", "B", "C")),
    "layer":  (3, ("X", "W", "B", "Y")),
    "network": (4, ("X", "W1", "B1", "W2", "B2", "Y")),
}


class ResultsError(Exception):
    pass


def parse_results(path, kind):
    """Return (dims, [{name, tag: array, ...}, ...]) for a results file."""
    n_dims, tags = RECORD_KINDS[kind]
    dims, cases, end_count = None, [], None

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            tag, values = fields[0], fields[1:]
            if tag == "dims":
                dims = tuple(int(v) for v in values)
            elif tag == "case":
                cases.append({"name": " ".join(values), "line": lineno})
            elif tag in tags and cases:
                cases[-1][tag] = np.array([int(v) for v in values], dtype=np.int64)
            elif tag == "end":
                end_count = int(values[0])
            else:
                raise ResultsError(f"{path}:{lineno}: unexpected line: {line.strip()[:60]}")

    if dims is None or len(dims) != n_dims:
        raise ResultsError(f"{path}: missing a 'dims' line with {n_dims} values")
    if end_count is None:
        raise ResultsError(f"{path}: no 'end' line; the simulation did not finish")
    if end_count != len(cases):
        raise ResultsError(f"{path}: 'end' says {end_count} cases, file has {len(cases)}")
    for case in cases:
        missing = [t for t in tags if t not in case]
        if missing:
            raise ResultsError(f"{path}:{case['line']}: case '{case['name']}' "
                               f"is missing {', '.join(missing)}")
    return dims, cases


def report(kind, mismatches, n_cases, n_values, shape_note):
    if mismatches:
        print(f"verify_results: FAIL - {len(mismatches)} of {n_values} values differ "
              f"from the NumPy golden model")
        for index, name, pos, rtl, ref in mismatches[:MAX_REPORTED]:
            print(f"  case {index} ({name}): {pos} RTL {rtl}, NumPy {ref}")
        return 1
    print(f"verify_results: PASS - {n_cases} {kind} runs, {n_values} values "
          f"({shape_note}) match the NumPy golden model")
    return 0


def verify_matmul(path):
    (m, k, n), cases = parse_results(path, "matmul")
    mismatches = []
    for index, case in enumerate(cases):
        c_rtl = case["C"].reshape(m, n)
        c_ref = matmul_int8(case["A"].reshape(m, k), case["B"].reshape(k, n))
        for i, j in np.argwhere(c_rtl != c_ref):
            mismatches.append((index, case["name"], f"C[{i}][{j}]",
                               int(c_rtl[i, j]), int(c_ref[i, j])))
    return report("multiply", mismatches, len(cases), len(cases) * m * n, f"M={m} K={k} N={n}")


def verify_layer(path):
    (in_size, out_size, relu), cases = parse_results(path, "layer")
    layer = layer_relu if relu else layer_affine
    mismatches = []
    for index, case in enumerate(cases):
        y_rtl = case["Y"]
        y_ref = layer(case["X"], case["W"].reshape(in_size, out_size), case["B"])
        for (j,) in np.argwhere(y_rtl != y_ref):
            mismatches.append((index, case["name"], f"Y[{j}]", int(y_rtl[j]), int(y_ref[j])))
    return report("layer", mismatches, len(cases), len(cases) * out_size,
                  f"INPUT_SIZE={in_size} OUTPUT_SIZE={out_size} "
                  f"{'with' if relu else 'without'} ReLU")


def verify_network(path):
    (in_size, hid_size, out_size, shift), cases = parse_results(path, "network")
    mismatches = []
    for index, case in enumerate(cases):
        y_rtl = case["Y"]
        y_ref = network_2layer(case["X"], case["W1"].reshape(in_size, hid_size), case["B1"],
                               case["W2"].reshape(hid_size, out_size), case["B2"], shift=shift)
        for (o,) in np.argwhere(y_rtl != y_ref):
            mismatches.append((index, case["name"], f"Y[{o}]", int(y_rtl[o]), int(y_ref[o])))
    return report("network", mismatches, len(cases), len(cases) * out_size,
                  f"{in_size}-{hid_size}-{out_size}, SHIFT={shift}")


def main(argv):
    if len(argv) != 3 or argv[1] not in RECORD_KINDS:
        print(f"usage: verify_results.py {{{'|'.join(RECORD_KINDS)}}} RESULTS_FILE", file=sys.stderr)
        return 2
    checkers = {"matmul": verify_matmul, "layer": verify_layer, "network": verify_network}
    try:
        return checkers[argv[1]](argv[2])
    except (OSError, ResultsError) as exc:
        print(f"verify_results: FAIL - {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
