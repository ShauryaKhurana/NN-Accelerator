# Verification

The rule this project was built under: **nothing is claimed to work until the
test has been run.** Every number in these docs came out of a simulator, and
where something is unverified it says so.

## Three independent things have to agree

```
   python/golden_model.py            tb/*_tb.sv                tb/*_vec_tb.sv
   (NumPy reference, with            (self-checking,           (vector-driven,
    its own self-test)                models timing)            Python-generated)
          |                                |                          |
          |  recomputes every              |  checks values AND       |  volume and
          |  value the RTL emits           |  cycle-exact timing      |  distribution
          v                                v                          v
   +--------------------------------------------------------------------------+
   |                      the same RTL, on two simulators                     |
   |              Verilator 5.052            Icarus Verilog 13.0              |
   +--------------------------------------------------------------------------+
```

The golden model is checked against plain Python before it is trusted to judge
the RTL (`make golden`): 72 matrix cases, 56 layer cases and 30 network cases
across four shift values, plus hand-computed values and range checks.

Two simulators matter more than it sounds. Icarus has caught things Verilator
accepts — a forward reference to a signal declared below the process using it,
an elaboration `$error` with format arguments, a ternary between two string
literals that silently pads to the wider one. Verilator has caught things
Icarus does not — the `--x-initial unique` plus `+verilator+rand+reset+2`
combination starts every flop at a random value, so a missing reset fails
instead of hiding behind a convenient zero.

## Flow 1: self-checking testbenches

One per RTL module. These generate their own stimulus, model the expected
behaviour, and check **timing as well as values** — exact busy-cycle counts
under random stalls, protocol rules on every clock edge, and resets from every
state.

| Testbench | Sections |
|---|---|
| `mac_tb.sv` | reset; directed products; accumulation and control; exhaustive operand sweep; accumulator range and wrap |
| `relu_tb.sv` | boundaries; walking-one / walking-zero; exhaustive |
| `requant_tb.sv` | boundaries; rounding (the shift floors, it does not truncate toward zero); walking one; exhaustive |
| `matmul_ctrl_tb.sv` | one weight load and one inference; stalls; reset from each busy state |
| `matrix_mult_tb.sv` | directed matrices; stalls, weight reuse, resets |
| `nn_layer_tb.sv` | directed; stalls and resets |
| `nn_accelerator_tb.sv` | directed; stalls and resets; several inferences in flight |

Four of them write a results file that `python/verify_results.py` recomputes
with NumPy, so the same run is checked twice: once for timing by the
testbench, once for arithmetic by Python.

What makes these more than smoke tests:

- **Exhaustive where the input space allows it.** ReLU at WIDTH=16 and the
  requantizer at 16->4 bits are checked over every possible input. The MAC
  sweeps every INT8 x INT8 product.
- **A reference model of the FSM.** `matmul_ctrl_tb.sv` models the transition
  table from `matmul_ctrl.sv` and compares state and every counter after each
  clock edge, not just the outputs.
- **Cycle accounting is exact.** Each job's busy cycles must equal the
  documented formula plus exactly one per stall cycle. This is what makes the
  performance numbers measurements rather than estimates.
- **Coverage bins fail the run.** A bin that is never exercised is an error,
  not a warning, so a test that stops reaching a case cannot pass quietly.
- **Documented overflow is tested, not avoided.** The MAC's wrap bound is
  checked at 131,071 terms; the layer is driven one past the largest safe bias
  to confirm it wraps exactly where the comment says.

## Flow 2: Python-driven vector regression

The inverse arrangement: Python generates the operands, the RTL consumes them,
Python checks what comes back. `make regression`.

```
  python/run_regression.py
      |  numpy: random INT8 operands + directed edge cases
      v
  vectors/regression_<dut>_<sim>.txt     (plain integers, read with $fscanf)
      |
      v
  tb/nn_accelerator_vec_tb.sv  |  tb/matrix_mult_vec_tb.sv
      |  one weight load per group, then one inference per case
      v
  sim/regression_<dut>_<sim>.results.txt
      |
      v
  python/run_regression.py  ->  numpy golden model  ->  exit 0 or nonzero
```

These testbenches invent nothing and check no timing. Their job is volume and
Python-chosen distributions. They do check the protocol (`m_last` position, no
X/Z on outputs, `done` a single-cycle pulse), range-check every vector value,
and guard every wait with a watchdog so a deadlock fails the run instead of
hanging it.

**Two DUTs, because they reach different things.** The network exercises the
whole datapath, but its hidden activations are ReLU'd and requantized to INT8,
so the accumulator's full width never reaches an output. The matrix multiplier
sits one level down, where every element of C leaves as a raw signed INT32 —
the same 1,000 cases check 64,640 values there against 10,150 logits at the
top.

**Edge cases are generated before the random ones:** all zeros, all 127, all
-128, -128 x 127, 127 x -128, the largest and most negative accumulations each
paired with the largest bias that still cannot wrap INT32, a checkerboard of
the extremes, small magnitudes around zero, an identity-like W1, and an index
pattern. One random group in four is drawn only from -128 and 127, so large
accumulations stay well represented instead of averaging out.

## What a full run measures

`make test` (Verilator) and `make test-iverilog` (Icarus), from a clean tree:

| | Verilator | Icarus |
|---|---|---|
| Testbench runs | 28 | 22 |
| NumPy checks | 15 | 15 |
| Values recomputed with NumPy, testbench flow | 53,531 over 2,758 runs | same |
| Vector-driven cases | 2,025 | 2,025 |
| Values recomputed with NumPy, regression flow | 74,790 | 74,790 |
| Errors | 0 | 0 |

Configurations covered: the matrix multiply at 8x8x8, 3x16x5 and 1x16x8 and at
1 and 4 MACs; the controller at 8x8x8, 1x1x1, 2x3x2 and 3x16x5 and at 1 and 4
MACs; the layer at 16->8, 4->3, 32->10 and 16->32, with and without ReLU, at 1
and 8 MACs; the network at 16-32-10 and 4-3-2, at SHIFT 8 and 0, and at 1, 4
and 8 MACs. `make regression-all` adds both simulators crossed with MAC widths
1, 4 and 8, with and without stalls, and the shapes 4-3-2 and 3x16x5.

For each shape the Verilator and Icarus results files are byte-identical.

## Mutation testing

The question a passing test suite cannot answer is whether it would notice a
bug. So at each phase, deliberate bugs were injected into the RTL and the
suite re-run. These were **one-off checks, not automated in the repo**:

| Phase | Injected | Caught |
|---|---|---|
| 1 — MAC | 9 | 9 |
| 2 — ReLU | 8 | 8 |
| 4 — controller FSM | 13 | 13 |
| 5 — layer | 7 | 7 |
| 6 — network | 10 | 10 |
| 7 — parallel MACs | 5 | 5 |
| 8 — weight-stationary | 10 | 9 |
| 9 — regression flow | 6 | 5 |

The two survivors are both worth more than the 66 catches:

**A 31-bit accumulator** was caught only by the MAC's range test at 131,071
terms — every ordinary dot product fits in 31 bits, so nothing else noticed.
That test exists because of it.

**The requantizer's arithmetic shift changed to a logical shift** survives
every network-level test, and is *unreachable* there rather than merely
missed: layer 1 applies ReLU, so the requantizer's input is never negative,
and `>>>` and `>>` agree on non-negative values. `requant_tb.sv` drives the
requantizer directly with negative values and catches it immediately. That is
the argument for keeping unit testbenches once a top-level one exists.

Some other findings worth recording:

- "Tail group treated as full" passes at 8x8x8, where N divides evenly by 4,
  and is caught at 3x16x5. That non-square shape is in the regression because
  of this.
- "Channel counter never resets" passes at a power-of-two OUTPUT_SIZE and is
  caught at 4->3.
- Four routing bugs in the network deadlock rather than miscompute, and are
  caught by the watchdog — which is why `send_beat` is guarded too. It was not
  at first, and the deadlock case hung the regression instead of failing it.

## Known gaps

- **No formal verification.** No property checks, no equivalence checking.
- **Mutation testing is not automated.** The scripts live outside the repo, so
  the numbers above are a record of what was run, not something CI reproduces.
- **Only two DUTs are vector-driven.** The MAC, ReLU, requantizer and layer
  are covered by their unit testbenches and by the network above them.
- **No gate-level or post-synthesis simulation.** Nothing has been through a
  synthesis tool, so there is no netlist to simulate and no timing to check.
- **The older testbenches are not lint-clean** under `verilator -Wall`; they
  use idioms it warns about. Only the two vector-driven ones are held to that
  standard, enforced by `make lint`.
- **Coverage is functional, hand-written bins.** There is no code or toggle
  coverage, and no coverage database.

## Running it

```sh
make venv            # .venv with numpy
make lint            # verilator -Wall on all RTL and the vector testbenches
make golden          # the NumPy model against plain Python
make test            # every testbench on Verilator, plus the NumPy checks
make test-iverilog   # the same on Icarus Verilog
make regression      # 1,000 randomized cases per DUT, Python-driven
make regression-all  # every simulator / MAC width / shape combination
```

Any failure exits nonzero. The testbenches print `TEST PASSED` or
`TEST FAILED` with a count, and the Makefile treats a missing `TEST PASSED` as
a failure even if the simulator exited 0.
