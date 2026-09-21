"""Check RTL results written by a testbench against the NumPy golden model.

    verify_results.py matmul RESULTS_FILE

RESULTS_FILE is written by tb/matrix_mult_tb.sv (+resultsfile=...):

    dims M K N
    case <name>
    A <M*K integers, row-major>
    B <K*N integers, row-major>
    C <M*N integers, row-major, read back from the DUT>
    ...
    end <number of cases>

Every C element is compared with golden_model.matmul_int8(A, B). Exits 0 if
all of them match, 1 on any mismatch or malformed/truncated file, and 2 on
bad usage.
"""

import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed: run `make venv` to create .venv with numpy")

from golden_model import matmul_int8

MAX_REPORTED = 10


class ResultsError(Exception):
    pass


def parse_matmul_results(path):
    """Return ((M, K, N), [(name, A, B, C), ...]) from a results file."""
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
            elif tag in ("A", "B", "C") and cases:
                cases[-1][tag] = np.array([int(v) for v in values], dtype=np.int64)
            elif tag == "end":
                end_count = int(values[0])
            else:
                raise ResultsError(f"{path}:{lineno}: unexpected line: {line.strip()[:60]}")

    if dims is None or len(dims) != 3:
        raise ResultsError(f"{path}: missing 'dims M K N' line")
    if end_count is None:
        raise ResultsError(f"{path}: no 'end' line; the simulation did not finish")
    if end_count != len(cases):
        raise ResultsError(f"{path}: 'end' says {end_count} cases, file has {len(cases)}")

    m, k, n = dims
    parsed = []
    for case in cases:
        try:
            a = case["A"].reshape(m, k)
            b = case["B"].reshape(k, n)
            c = case["C"].reshape(m, n)
        except (KeyError, ValueError):
            raise ResultsError(f"{path}:{case['line']}: case '{case['name']}' is incomplete")
        parsed.append((case["name"], a, b, c))
    return dims, parsed


def verify_matmul(path):
    (m, k, n), cases = parse_matmul_results(path)
    mismatches = []
    for index, (name, a, b, c_rtl) in enumerate(cases):
        c_ref = matmul_int8(a, b)
        for i, j in np.argwhere(c_rtl != c_ref):
            mismatches.append((index, name, i, j, int(c_rtl[i, j]), int(c_ref[i, j])))

    elements = len(cases) * m * n
    if mismatches:
        print(f"verify_results: FAIL - {len(mismatches)} of {elements} elements differ "
              f"from the NumPy golden model")
        for index, name, i, j, rtl, ref in mismatches[:MAX_REPORTED]:
            print(f"  case {index} ({name}): C[{i}][{j}] RTL {rtl}, NumPy {ref}")
        return 1
    print(f"verify_results: PASS - {len(cases)} multiplies, {elements} elements "
          f"(M={m} K={k} N={n}) match the NumPy golden model")
    return 0


def main(argv):
    if len(argv) != 3 or argv[1] != "matmul":
        print("usage: verify_results.py matmul RESULTS_FILE", file=sys.stderr)
        return 2
    try:
        return verify_matmul(argv[2])
    except (OSError, ResultsError) as exc:
        print(f"verify_results: FAIL - {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
