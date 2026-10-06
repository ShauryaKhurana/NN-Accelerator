"""Python-driven random regression for the accelerator.

    run_regression.py [--dut both|network|matmul] [--cases N]
                      [--sim verilator|iverilog] [--num-macs P]
                      [--seed S] [--stalls] [--keep]

Python owns the whole loop:

  1. generate random INT8 activations and weights, plus directed edge cases
  2. compute the expected outputs with NumPy (python/golden_model.py)
  3. write a vector file
  4. build and run the SystemVerilog simulation
  5. parse what the RTL produced
  6. compare it against NumPy
  7. report every mismatch

Exits 0 only if every simulation ran to completion and every value matched.
Any mismatch, protocol error, deadlock, build failure or crash exits nonzero.

Two DUTs, because they see different things:

  network  tb/nn_accelerator_vec_tb.sv - the whole datapath end to end, but
           the hidden activations pass through ReLU and are requantized to
           INT8, so the full accumulator width is never visible at the output.
  matmul   tb/matrix_mult_vec_tb.sv - one level down, where every element of C
           leaves as a raw signed INT32, so large negative accumulations are
           checked directly.

Cases are grouped: weights are loaded once per group and reused by the
inferences in it, which is what the hardware is built for since Phase 8.
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed: run `make venv` to create .venv with numpy")

from golden_model import matmul_int8, network_2layer
from verify_results import ResultsError, parse_results

ROOT = Path(__file__).resolve().parent.parent
NET_RTL = [
    "rtl/matmul_pkg.sv", "rtl/mac.sv", "rtl/relu.sv", "rtl/requant.sv",
    "rtl/matmul_ctrl.sv", "rtl/matrix_mult.sv", "rtl/nn_layer.sv", "rtl/nn_accelerator.sv",
]
MM_RTL = ["rtl/matmul_pkg.sv", "rtl/mac.sv", "rtl/matmul_ctrl.sv", "rtl/matrix_mult.sv"]

INT8_MIN, INT8_MAX = -128, 127
MAX_REPORTED = 10
PER_GROUP = 8          # inferences sharing one weight load


def bias_limit(terms):
    """Largest |bias| that cannot make an INT32 accumulator wrap.

    A dot product of `terms` INT8 products lies in [-terms*16256, terms*16384],
    so this much headroom is always left.
    """
    return (1 << 31) - 1 - terms * 16384


def extremes(rng, shape):
    return rng.choice([INT8_MIN, INT8_MAX], size=shape).astype(np.int64)


# =============================================================================
# network
# =============================================================================
def net_groups(rng, shape, n_cases):
    """The directed edge cases, plus n_cases randomized inferences on top."""
    in_size, hid, out = shape
    groups, labels = [], []

    def add(group, label):
        groups.append(group)
        labels.extend(f"{label}[{i}]" for i in range(len(group["xs"])))

    def const(x_val, w_val, b_val, n=1):
        return {"w1": np.full((in_size, hid), w_val, dtype=np.int64),
                "b1": np.full(hid, b_val, dtype=np.int64),
                "w2": np.full((hid, out), w_val, dtype=np.int64),
                "b2": np.full(out, b_val, dtype=np.int64),
                "xs": [np.full(in_size, x_val, dtype=np.int64) for _ in range(n)]}

    add(const(0, 0, 0), "all zeros")
    add(const(INT8_MAX, INT8_MAX, 0), "all 127")
    add(const(INT8_MIN, INT8_MIN, 0), "all -128")
    add(const(INT8_MIN, INT8_MAX, 0), "-128 x 127")
    add(const(INT8_MAX, INT8_MIN, 0), "127 x -128")
    # -128 * -128 = +16384 is the largest single product, so these two sit at
    # the edge of what the INT32 accumulator holds without wrapping.
    add(const(INT8_MIN, INT8_MIN, bias_limit(in_size)), "largest accumulation + largest safe bias")
    add(const(INT8_MIN, INT8_MAX, -bias_limit(in_size)), "most negative accumulation + safe bias")

    # Mixed signs in every dot product
    w1 = np.where((np.add.outer(np.arange(in_size), np.arange(hid)) % 2) == 0,
                  INT8_MAX, INT8_MIN).astype(np.int64)
    w2 = np.where((np.add.outer(np.arange(hid), np.arange(out)) % 2) == 0,
                  INT8_MIN, INT8_MAX).astype(np.int64)
    add({"w1": w1, "b1": np.zeros(hid, dtype=np.int64),
         "w2": w2, "b2": np.zeros(out, dtype=np.int64),
         "xs": [np.where(np.arange(in_size) % 2 == 0, INT8_MAX, INT8_MIN).astype(np.int64),
                np.where(np.arange(in_size) % 2 == 0, INT8_MIN, INT8_MAX).astype(np.int64)]},
        "checkerboard of extremes")

    # Small magnitudes: the requantizer around zero
    add({"w1": rng.integers(-2, 3, size=(in_size, hid), dtype=np.int64),
         "b1": np.zeros(hid, dtype=np.int64),
         "w2": rng.integers(-2, 3, size=(hid, out), dtype=np.int64),
         "b2": np.zeros(out, dtype=np.int64),
         "xs": [rng.integers(-2, 3, size=in_size, dtype=np.int64) for _ in range(4)]},
        "small magnitudes")

    # Identity-like W1: each input routed to one hidden channel
    w1 = np.zeros((in_size, hid), dtype=np.int64)
    for k in range(in_size):
        w1[k, k % hid] = 1
    add({"w1": w1, "b1": np.zeros(hid, dtype=np.int64),
         "w2": rng.integers(INT8_MIN, INT8_MAX + 1, size=(hid, out), dtype=np.int64),
         "b2": np.zeros(out, dtype=np.int64),
         "xs": [rng.integers(INT8_MIN, INT8_MAX + 1, size=in_size, dtype=np.int64)
                for _ in range(2)]},
        "identity-ish W1")

    g, n_random = 0, 0
    while n_random < n_cases:
        want = min(PER_GROUP, n_cases - n_random)
        b1_lim = min(bias_limit(in_size), 1 << 20)
        b2_lim = min(bias_limit(hid), 1 << 20)
        group = {"b1": rng.integers(-b1_lim, b1_lim + 1, size=hid, dtype=np.int64),
                 "b2": rng.integers(-b2_lim, b2_lim + 1, size=out, dtype=np.int64)}
        if g % 4 == 3:          # keep large accumulations well represented
            group["w1"] = extremes(rng, (in_size, hid))
            group["w2"] = extremes(rng, (hid, out))
            group["xs"] = [extremes(rng, in_size) for _ in range(want)]
            add(group, f"random extremes {g}")
        else:
            group["w1"] = rng.integers(INT8_MIN, INT8_MAX + 1, size=(in_size, hid), dtype=np.int64)
            group["w2"] = rng.integers(INT8_MIN, INT8_MAX + 1, size=(hid, out), dtype=np.int64)
            group["xs"] = [rng.integers(INT8_MIN, INT8_MAX + 1, size=in_size, dtype=np.int64)
                           for _ in range(want)]
            add(group, f"random {g}")
        n_random += want
        g += 1
    return groups, labels, n_random


def net_write(path, shape, shift, groups):
    in_size, hid, out = shape
    with open(path, "w") as f:
        f.write(f"{in_size} {hid} {out} {shift} {len(groups)}\n")
        for group in groups:
            f.write(f"{len(group['xs'])}\n")
            for name in ("w1", "b1", "w2", "b2"):
                f.write(" ".join(str(int(v)) for v in np.ravel(group[name])) + "\n")
            for x in group["xs"]:
                f.write(" ".join(str(int(v)) for v in x) + "\n")


def net_check(cases, shape, shift, labels):
    in_size, hid, out = shape
    mismatches = []
    for index, (case, label) in enumerate(zip(cases, labels)):
        ref = network_2layer(case["X"], case["W1"].reshape(in_size, hid), case["B1"],
                             case["W2"].reshape(hid, out), case["B2"], shift=shift)
        for (o,) in np.argwhere(case["Y"] != ref):
            mismatches.append((index, label, f"Y[{o}]", int(case["Y"][o]), int(ref[o])))
    return mismatches, len(cases) * out


# =============================================================================
# matmul
# =============================================================================
def mm_groups(rng, shape, n_cases):
    """The directed edge cases, plus n_cases randomized multiplies on top."""
    m, k, n = shape
    groups, labels = [], []

    def add(group, label):
        groups.append(group)
        labels.extend(f"{label}[{i}]" for i in range(len(group["as"])))

    def const(a_val, b_val, count=1):
        return {"b": np.full((k, n), b_val, dtype=np.int64),
                "as": [np.full((m, k), a_val, dtype=np.int64) for _ in range(count)]}

    add(const(0, 0), "all zeros")
    add(const(INT8_MAX, INT8_MAX), "all 127")
    add(const(INT8_MIN, INT8_MIN), "all -128")          # every element = K*16384
    add(const(INT8_MIN, INT8_MAX), "-128 x 127")        # every element = K*-16256
    add(const(INT8_MAX, INT8_MIN), "127 x -128")

    # Mixed signs in every dot product
    a = np.where((np.add.outer(np.arange(m), np.arange(k)) % 2) == 0,
                 INT8_MAX, INT8_MIN).astype(np.int64)
    b = np.where((np.add.outer(np.arange(k), np.arange(n)) % 2) == 0,
                 INT8_MIN, INT8_MAX).astype(np.int64)
    add({"b": b, "as": [a, -a - 1]}, "checkerboard of extremes")

    # Identity B: C must be A widened, which catches misrouted indices
    if k == n:
        add({"b": np.eye(k, dtype=np.int64),
             "as": [rng.integers(INT8_MIN, INT8_MAX + 1, size=(m, k), dtype=np.int64)
                    for _ in range(2)]}, "identity B")

    # Distinct values: a swapped index shows up immediately
    add({"b": ((127 - 3 * np.arange(k * n).reshape(k, n)) % 256 - 128).astype(np.int64),
         "as": [((4 * np.arange(m * k).reshape(m, k)) % 256 - 128).astype(np.int64)]},
        "index pattern")

    g, n_random = 0, 0
    while n_random < n_cases:
        want = min(PER_GROUP, n_cases - n_random)
        if g % 3 == 2:          # extremes only: the largest accumulations
            add({"b": extremes(rng, (k, n)),
                 "as": [extremes(rng, (m, k)) for _ in range(want)]}, f"random extremes {g}")
        else:
            add({"b": rng.integers(INT8_MIN, INT8_MAX + 1, size=(k, n), dtype=np.int64),
                 "as": [rng.integers(INT8_MIN, INT8_MAX + 1, size=(m, k), dtype=np.int64)
                        for _ in range(want)]}, f"random {g}")
        n_random += want
        g += 1
    return groups, labels, n_random


def mm_write(path, shape, _shift, groups):
    m, k, n = shape
    with open(path, "w") as f:
        f.write(f"{m} {k} {n} {len(groups)}\n")
        for group in groups:
            f.write(f"{len(group['as'])}\n")
            f.write(" ".join(str(int(v)) for v in np.ravel(group["b"])) + "\n")
            for a in group["as"]:
                f.write(" ".join(str(int(v)) for v in np.ravel(a)) + "\n")


def mm_check(cases, shape, _shift, labels):
    m, k, n = shape
    mismatches = []
    for index, (case, label) in enumerate(zip(cases, labels)):
        got = case["C"].reshape(m, n)
        ref = matmul_int8(case["A"].reshape(m, k), case["B"].reshape(k, n))
        for i, j in np.argwhere(got != ref):
            mismatches.append((index, label, f"C[{i}][{j}]", int(got[i, j]), int(ref[i, j])))
    return mismatches, len(cases) * m * n


# =============================================================================
DUTS = {
    "network": {
        "top": "nn_accelerator_vec_tb", "tb": "tb/nn_accelerator_vec_tb.sv", "rtl": NET_RTL,
        "kind": "network", "shape": (16, 32, 10), "unit": "inferences",
        "params": lambda shape, shift: dict(INPUT_SIZE=shape[0], HIDDEN_SIZE=shape[1],
                                            OUTPUT_SIZE=shape[2], SHIFT=shift),
        "dims": lambda shape, shift: (shape[0], shape[1], shape[2], shift),
        "groups": net_groups, "write": net_write, "check": net_check,
    },
    "matmul": {
        "top": "matrix_mult_vec_tb", "tb": "tb/matrix_mult_vec_tb.sv", "rtl": MM_RTL,
        "kind": "matmul", "shape": (8, 8, 8), "unit": "multiplies",
        "params": lambda shape, _shift: dict(M=shape[0], K=shape[1], N=shape[2]),
        "dims": lambda shape, _shift: shape,
        "groups": mm_groups, "write": mm_write, "check": mm_check,
    },
}


def run(cmd, what, quiet):
    if not quiet:
        print(f"  $ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"regression: FAIL - {what} exited {proc.returncode}")
        for line in (proc.stdout + proc.stderr).strip().splitlines()[-25:]:
            print(f"    {line}")
        return None
    return proc.stdout


def lz4_flags():
    """Verilator's FST writer includes <lz4.h>, which Homebrew does not put on
    the default include path."""
    try:
        prefix = subprocess.run(["brew", "--prefix", "lz4"], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        return []
    if prefix and (Path(prefix) / "include" / "lz4.h").exists():
        return ["-CFLAGS", f"-I{prefix}/include", "-LDFLAGS", f"-L{prefix}/lib"]
    return []


def simulate(dut, sim, shape, shift, num_macs, vectors, results, seed, stalls, quiet,
             waves=None, max_cases=0):
    top = dut["top"]
    # A traced build is a different binary, so keep it out of the normal one's
    # directory rather than forcing a rebuild on every switch.
    build = ROOT / "sim" / sim / (top + ("_waves" if waves else ""))
    build.mkdir(parents=True, exist_ok=True)
    params = dict(dut["params"](shape, shift), NUM_MACS=num_macs)

    if sim == "verilator":
        cmd = ["verilator", "--binary", "-j", "0", "--x-initial", "unique", "--quiet",
               "--unroll-count", "1", "-CFLAGS", "-Wno-unknown-warning-option",
               "--top-module", top, "--Mdir", str(build)]
        if waves:
            if waves.suffix == ".fst":
                cmd += ["--trace-fst"] + lz4_flags()
            else:
                cmd += ["--trace"]
        cmd += [f"-G{k}={v}" for k, v in params.items()] + dut["rtl"] + [dut["tb"]]
        if run(cmd, f"{sim} build", quiet) is None:
            return None
        sim_cmd = [str(build / f"V{top}"), "+verilator+rand+reset+2", f"+verilator+seed+{seed}"]
    else:
        vvp = build / f"{top}.vvp"
        cmd = ["iverilog", "-g2012", "-Wall", "-o", str(vvp), "-s", top]
        cmd += [f"-P{top}.{k}={v}" for k, v in params.items()] + dut["rtl"] + [dut["tb"]]
        if run(cmd, f"{sim} build", quiet) is None:
            return None
        sim_cmd = ["vvp", "-n", str(vvp)]

    sim_cmd += [f"+seed={seed}", f"+vectors={vectors}", f"+resultsfile={results}"]
    if stalls:
        sim_cmd.append("+stalls")
    if waves:
        sim_cmd.append(f"+dumpfile={waves}")
    if max_cases:
        sim_cmd.append(f"+maxcases={max_cases}")
    return run(sim_cmd, "simulation", quiet)


def regress(name, args):
    """Run one DUT end to end. Returns True if everything matched."""
    dut = DUTS[name]
    shape = tuple(int(v) for v in args.shape.split(",")) if args.shape else dut["shape"]
    if len(shape) != 3 or any(v < 1 for v in shape):
        print("regression: FAIL - --shape needs three positive integers", file=sys.stderr)
        return False

    vec_dir = ROOT / "vectors"
    res_dir = ROOT / "sim"
    vec_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)
    vectors = vec_dir / f"regression_{name}_{args.sim}.txt"
    results = res_dir / f"regression_{name}_{args.sim}.results.txt"
    results.unlink(missing_ok=True)          # never check a stale file

    rng = np.random.default_rng(args.seed)
    groups, labels, n_random = dut["groups"](rng, shape, args.cases)
    dut["write"](vectors, shape, args.shift, groups)

    shape_note = "-".join(str(v) for v in shape)
    print(f"regression [{name}]: {len(labels)} {dut['unit']} "
          f"({n_random} randomized + {len(labels) - n_random} directed) "
          f"over {len(groups)} weight groups, "
          f"{shape_note} NUM_MACS={args.num_macs} on {args.sim}, seed {args.seed}"
          f"{', with stalls' if args.stalls else ''}")
    if args.keep:
        print(f"  vectors {vectors.relative_to(ROOT)}  results {results.relative_to(ROOT)}")

    waves = None
    if args.waves:
        wave_dir = ROOT / "waveforms"
        wave_dir.mkdir(exist_ok=True)
        waves = wave_dir / f"regression_{name}.{args.waves}"
    stdout = simulate(dut, args.sim, shape, args.shift, args.num_macs,
                      vectors, results, args.seed, args.stalls, args.quiet,
                      waves, args.max_cases)
    if stdout is None:
        return False
    for line in stdout.splitlines():
        if line.startswith(("TEST ", "ERROR")):
            print(f"  {line}")
    if "TEST PASSED" not in stdout:
        print(f"regression [{name}]: FAIL - the simulation did not report TEST PASSED")
        return False

    try:
        dims, cases = parse_results(results, dut["kind"])
        if dims != dut["dims"](shape, args.shift):
            raise ResultsError(f"{results}: dims {dims}, expected {dut['dims'](shape, args.shift)}")
        expected = min(len(labels), args.max_cases) if args.max_cases else len(labels)
        if len(cases) != expected:
            raise ResultsError(f"{results}: {len(cases)} cases, expected {expected}")
        labels = labels[:expected]
        mismatches, n_values = dut["check"](cases, shape, args.shift, labels)
    except (OSError, ResultsError) as exc:
        print(f"regression [{name}]: FAIL - {exc}")
        return False

    if mismatches:
        print(f"regression [{name}]: FAIL - {len(mismatches)} of {n_values} values "
              f"differ from NumPy")
        for index, label, pos, got, ref in mismatches[:MAX_REPORTED]:
            print(f"  case {index} ({label}): {pos} RTL {got}, NumPy {ref}")
        if len(mismatches) > MAX_REPORTED:
            print(f"  ... and {len(mismatches) - MAX_REPORTED} more")
        return False

    if not args.keep:
        vectors.unlink(missing_ok=True)
        results.unlink(missing_ok=True)
    if waves:
        print(f"  wrote {waves.relative_to(ROOT)}")
    print(f"regression [{name}]: PASS - {len(labels)} {dut['unit']}"
          f"{'' if args.max_cases else f' ({n_random} randomized)'}, "
          f"{n_values} values match NumPy")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dut", choices=("both", "network", "matmul"), default="both")
    ap.add_argument("--cases", type=int, default=1000,
                    help="randomized cases per DUT, on top of the directed ones (default 1000)")
    ap.add_argument("--sim", choices=("verilator", "iverilog"), default="verilator")
    ap.add_argument("--num-macs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shape", default=None, help="three ints, e.g. 16,32,10 or 8,8,8")
    ap.add_argument("--shift", type=int, default=8, help="requantization shift (network only)")
    ap.add_argument("--stalls", action="store_true",
                    help="drive random input gaps and output backpressure")
    ap.add_argument("--keep", action="store_true", help="keep the vector and results files")
    ap.add_argument("--quiet", action="store_true", help="do not echo the build commands")
    ap.add_argument("--waves", choices=("vcd", "fst"), default=None,
                    help="write waveforms/regression_<dut>.<fmt>; pair with --max-cases")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="stop the simulation after n cases (0 = all); for small traces")
    args = ap.parse_args(argv)

    if args.cases < 1:
        print("regression: FAIL - --cases must be at least 1", file=sys.stderr)
        return 2
    names = list(DUTS) if args.dut == "both" else [args.dut]
    if args.shape and len(names) > 1:
        print("regression: FAIL - --shape applies to one DUT; pass --dut too", file=sys.stderr)
        return 2

    ok = True
    for name in names:
        if not regress(name, args):
            ok = False
    print(f"regression: {'PASS - every DUT matched NumPy' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
