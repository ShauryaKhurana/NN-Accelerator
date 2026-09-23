"""NumPy golden model for the INT8 neural-network accelerator.

These functions define the arithmetic the RTL must reproduce exactly: signed
INT8 operands, exact products, and INT32 accumulation. They are written with
ordinary NumPy array operations rather than mirroring the hardware's
structure, so they serve as an independent reference.

Run this file directly to self-test the model against plain Python integer
arithmetic.
"""

import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed: run `make venv` to create .venv with numpy")

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def _as_int8_matrix(name, m):
    m = np.asarray(m)
    if m.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {m.shape}")
    if not np.issubdtype(m.dtype, np.integer):
        raise TypeError(f"{name} must contain integers, got {m.dtype}")
    if m.size and (m.min() < INT8_MIN or m.max() > INT8_MAX):
        raise ValueError(f"{name} has values outside INT8 [{INT8_MIN}, {INT8_MAX}]")
    return m


def matmul_int8(a, b):
    """Return C = A @ B for signed INT8 matrices as an INT32 matrix.

    The multiply runs in int64 because NumPy multiplies int8 arrays in int8
    and would wrap. The result is then checked against the INT32 range of the
    hardware accumulator before it is narrowed.
    """
    a = _as_int8_matrix("A", a)
    b = _as_int8_matrix("B", b)
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"cannot multiply shapes {a.shape} and {b.shape}")
    c = a.astype(np.int64) @ b.astype(np.int64)
    if c.size and (c.min() < INT32_MIN or c.max() > INT32_MAX):
        raise OverflowError("result does not fit the INT32 accumulator")
    return c.astype(np.int32)


def relu_int32(x):
    """Return ReLU(x) for INT32 values: negatives become 0."""
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"ReLU input must contain integers, got {x.dtype}")
    if x.size and (x.min() < INT32_MIN or x.max() > INT32_MAX):
        raise ValueError("ReLU input outside INT32")
    return np.maximum(x, 0).astype(np.int32)


def layer_relu(x, w, b):
    """Return Y = ReLU(X @ W + B) for one fully connected INT8 layer.

    X is an INPUT_SIZE vector of INT8 activations, W is INPUT_SIZE x
    OUTPUT_SIZE INT8 weights, and B is one INT32 bias per output. The result
    is OUTPUT_SIZE INT32 values. Both the dot products and the bias add must
    fit INT32, which is what the hardware accumulator holds; anything larger
    raises rather than wrapping silently.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"X must be a 1-D vector, got shape {x.shape}")
    dot = matmul_int8(x.reshape(1, -1), w)[0]

    b = np.asarray(b)
    if b.ndim != 1 or b.shape[0] != dot.shape[0]:
        raise ValueError(f"B must have {dot.shape[0]} elements, got shape {b.shape}")
    if not np.issubdtype(b.dtype, np.integer):
        raise TypeError(f"B must contain integers, got {b.dtype}")
    if b.size and (b.min() < INT32_MIN or b.max() > INT32_MAX):
        raise ValueError("B has values outside INT32")

    total = dot.astype(np.int64) + b.astype(np.int64)
    if total.size and (total.min() < INT32_MIN or total.max() > INT32_MAX):
        raise OverflowError("X @ W + B does not fit the INT32 accumulator")
    return relu_int32(total.astype(np.int32))


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def _matmul_reference(a, b):
    """Plain-Python triple loop on arbitrary-precision ints."""
    a, b = np.asarray(a).tolist(), np.asarray(b).tolist()
    inner, cols = len(b), len(b[0])
    return [[sum(row[k] * b[k][j] for k in range(inner)) for j in range(cols)] for row in a]


def _expect(condition, message):
    if not condition:
        raise AssertionError(message)


def self_test(seed=1):
    rng = np.random.default_rng(seed)
    cases = 0
    for m, k, n in [(8, 8, 8), (3, 16, 5), (1, 16, 8)]:
        tests = [
            ("zeros", np.zeros((m, k), int), np.zeros((k, n), int)),
            ("all 127", np.full((m, k), 127), np.full((k, n), 127)),
            ("all -128", np.full((m, k), -128), np.full((k, n), -128)),
            ("-128 x 127", np.full((m, k), -128), np.full((k, n), 127)),
        ]
        tests += [(f"random {r}", rng.integers(-128, 128, (m, k)), rng.integers(-128, 128, (k, n)))
                  for r in range(20)]
        for name, a, b in tests:
            got = matmul_int8(a, b)
            _expect(got.dtype == np.int32, f"{m}x{k}x{n} {name}: dtype {got.dtype}, expected int32")
            _expect(got.tolist() == _matmul_reference(a, b), f"{m}x{k}x{n} {name}: wrong result")
            cases += 1

    # Hand-computed values for an 8-term dot product
    for value_a, value_b, expected in [(127, 127, 8 * 16129),      #  129,032
                                       (-128, -128, 8 * 16384),    #  131,072: largest
                                       (-128, 127, 8 * -16256)]:   # -130,048: most negative
        c = matmul_int8(np.full((8, 8), value_a), np.full((8, 8), value_b))
        _expect(bool((c == expected).all()), f"all {value_a} x all {value_b}: expected {expected}")

    # Out-of-range operands must be rejected, not silently wrapped
    for bad in (np.full((2, 2), 128), np.full((2, 2), -129)):
        try:
            matmul_int8(bad, bad)
        except ValueError:
            continue
        raise AssertionError(f"value {bad[0, 0]} was accepted as INT8")

    # Layer: Y = ReLU(X @ W + B), against the same plain-Python arithmetic
    layer_cases = 0
    for in_size, out_size in [(16, 8), (16, 32), (4, 3), (1, 1)]:
        tests = [
            ("zeros", np.zeros(in_size, int), np.zeros((in_size, out_size), int),
             np.zeros(out_size, int)),
            ("bias only", np.zeros(in_size, int), np.zeros((in_size, out_size), int),
             rng.integers(-1000, 1000, out_size)),
            ("all 127", np.full(in_size, 127), np.full((in_size, out_size), 127),
             np.zeros(out_size, int)),
            ("negative sums", np.full(in_size, -128), np.full((in_size, out_size), 127),
             np.zeros(out_size, int)),
        ]
        tests += [(f"random {r}", rng.integers(-128, 128, in_size),
                   rng.integers(-128, 128, (in_size, out_size)),
                   rng.integers(-300000, 300000, out_size)) for r in range(10)]
        for name, x, w, b in tests:
            got = layer_relu(x, w, b)
            want = [max(0, sum(int(x[k]) * int(w[k][j]) for k in range(in_size)) + int(b[j]))
                    for j in range(out_size)]
            _expect(got.dtype == np.int32, f"layer {in_size}x{out_size} {name}: dtype {got.dtype}")
            _expect(got.tolist() == want, f"layer {in_size}x{out_size} {name}: wrong result")
            layer_cases += 1

    # Hand-computed: 16 inputs of 127 x 127 sum to 258,064; the same with
    # -128 x 127 sums to -260,096 and ReLU clamps it to 0
    _expect(int(layer_relu(np.full(16, 127), np.full((16, 4), 127), np.zeros(4, int))[0]) == 258064,
            "16 x (127*127) should be 258064")
    _expect(int(layer_relu(np.full(16, -128), np.full((16, 4), 127), np.zeros(4, int))[0]) == 0,
            "a negative sum should clamp to 0")

    # A bias that pushes the sum past INT32 must raise, not wrap
    try:
        layer_relu(np.full(16, 127), np.full((16, 1), 127), np.array([INT32_MAX]))
    except OverflowError:
        pass
    else:
        raise AssertionError("a bias overflowing INT32 was accepted")

    print(f"golden_model self-test: PASS ({cases} matrix cases, {layer_cases} layer cases "
          f"vs. plain Python, hand-computed values, input range checks)")


if __name__ == "__main__":
    self_test()
