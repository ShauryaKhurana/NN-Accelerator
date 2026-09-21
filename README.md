# NN-Accelerator

An INT8 neural-network accelerator in synthesizable SystemVerilog, being built to
run entirely in simulation and to be verified against a Python/NumPy golden
model. No FPGA required.

The project is built incrementally; each phase must pass its tests before the
next one starts.

## Status

| Phase | Block                          | Status      |
|-------|--------------------------------|-------------|
| 1     | Signed INT8 MAC unit           | done — verified on Verilator and Icarus |
| 2     | ReLU                           | done — verified on Verilator and Icarus |
| 3     | 8×8 matrix multiply            | done — verified on Verilator and Icarus, and against NumPy |
| 4     | Control FSM + handshaking      | not started |
| 5     | NN layer `ReLU(X·W + B)`       | not started |
| 6     | Two-layer network (16→32→10)   | not started |
| 7     | Parallel MAC array             | not started |
| 8     | Pipelining                     | not started |
| 9     | Python-driven random regression| not started |
| 10    | Waveform tooling               | not started |
| 11    | Performance report             | not started |
| 12    | Documentation                  | not started |

## Repository layout

```
rtl/        synthesizable SystemVerilog
tb/         SystemVerilog testbenches
python/     NumPy golden model and RTL result checking
sim/        simulator build products and logs (generated, git-ignored)
vectors/    generated test vectors (git-ignored)
waveforms/  VCD/FST dumps (git-ignored)
docs/       architecture / verification / performance notes
```

## Prerequisites (macOS, Apple Silicon)

```sh
brew install verilator        # primary simulator (5.x, needs Xcode CLT for C++)
brew install icarus-verilog   # secondary simulator for cross-checking
brew install surfer           # waveform viewer (the GTKWave cask was disabled in 2025)
```

Python dependencies (NumPy, listed in `requirements.txt`). Homebrew's Python
refuses global `pip install` (PEP 668), so the project uses a virtual
environment, and the Makefile picks up `.venv` automatically:

```sh
make venv            # python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Check what is installed with `make check-tools`.

## Running the tests

```sh
make test            # Verilator: RTL lint + all testbenches (primary)
make test-iverilog   # Icarus Verilog: the same testbenches (cross-check)
make mac SEED=1234   # one testbench with a different random seed
make mac-waves       # short run that writes waveforms/mac_tb.vcd (also relu-, matmul-waves)
make help            # list every target
```

Every testbench prints a single `TEST PASSED` / `TEST FAILED` line. A failure
exits nonzero, so `make` stops. Logs go to `sim/<run>.<simulator>.log`.

To view a waveform: `surfer waveforms/mac_tb.vcd` (GTKWave also reads VCD if
you have it). Waveform runs record only a testbench's directed sections, which
keeps the files small: about 20 KB for the MAC and ReLU, and 1.4 MB for the
matrix multiply. Add `+dumpall` to record everything.

## Phase 1: MAC unit (`rtl/mac.sv`)

A parameterized signed multiply-accumulate (`DATA_WIDTH=8`, `ACC_WIDTH=32`)
that performs one multiply-accumulate per clock: `acc <= acc + a*b`.

| rst | clear | en | acc after the next rising edge            |
|-----|-------|----|-------------------------------------------|
| 1   | –     | –  | 0 (synchronous, active-high reset)        |
| 0   | 1     | 1  | `a*b`: starts a new sum, no idle cycle    |
| 0   | 1     | 0  | 0                                         |
| 0   | 0     | 1  | `acc + a*b`                               |
| 0   | 0     | 0  | `acc` (hold; `a`, `b` ignored)            |

- Operands are sign-extended explicitly, and the 16-bit product is exact
  (the extreme case is `-128 * -128 = +16384`).
- The accumulator wraps modulo 2^32 (two's complement), with no saturation.
  No sum of 131,071 or fewer INT8 products can wrap. In general the bound is
  `2^(ACC_WIDTH - 2*DATA_WIDTH + 1) - 1` terms.
- Latency is 1 clock: combinational multiply and add into a single register.
  Pipelining comes in Phase 8.

**Verification (`tb/mac_tb.sv`).** The testbench is self-checking. Every cycle
it compares the DUT against an independent 64-bit reference model. Directed
tests also check values worked out by hand.

| Section            | What it covers                                                    |
|--------------------|-------------------------------------------------------------------|
| Reset              | random/X initial state cleared; reset beats `en`/`clear`; reset mid-sum |
| Directed products  | every sign combination, zeros, INT8 extremes, `-128 * -1 = +128`  |
| Accumulation       | multi-cycle sums, hold, clear, clear+load, back-to-back dot products |
| Exhaustive         | all 65,536 INT8 × INT8 operand pairs                              |
| Accumulator range  | largest positive / most negative 131,071-term sums, and the wrap one term later |
| Constrained random | 20,000 cycles of random `rst`/`clear`/`en`/operands, biased toward corner values |
| Coverage           | 12 behavior bins (control modes, sign classes, MIN×MIN, wrap); each must be hit |

Measured results (seed 1): 347,723 cycles and 347,754 checks with 0 errors on
both Verilator 5.052 and Icarus 13.0. A second parameterization
(INT4 × INT4 → 12-bit accumulator) also passes on both.

## Phase 2: ReLU (`rtl/relu.sv`)

`y = (x < 0) ? 0 : x` for signed `WIDTH`-bit values (`WIDTH=32` by default).
The module is purely combinational.

- A two's-complement value is negative exactly when its sign bit is set, so
  the sign bit selects between `x` and 0. No comparator is needed. At gate
  level this is one AND gate per bit: `y[i] = x[i] & ~x[WIDTH-1]`.
- The output is never negative. It keeps the input's signed type so it can
  feed signed arithmetic directly.

**Verification (`tb/relu_tb.sv`).** Each check drives `x`, waits 1 ns, and
compares `y` with a 64-bit reference model. The model uses a signed
comparison, while the RTL uses the sign bit.

| Section      | What it covers                                                    |
|--------------|-------------------------------------------------------------------|
| Boundaries   | 0, ±1, ±2, ±100, MAX, MAX−1, MIN, MIN+1                            |
| Bit patterns | walking-one and walking-zero with the sign bit clear (passes through) and set (becomes 0), for every magnitude bit |
| Exhaustive   | every input value when `WIDTH ≤ 16` (all 65,536 for `WIDTH=16`)   |
| Random       | 100,000 values: uniform, near zero, and near MAX/MIN              |
| Coverage     | every input bit seen at 0 and 1; every output magnitude bit seen at 1; output sign bit never set |

Measured results (seed 1), with 0 errors on both Verilator 5.052 and Icarus 13.0:

- `WIDTH=32`: 100,135 checks.
- `WIDTH=16`: 165,607 checks, including the exhaustive sweep.

## Phase 3: matrix multiply (`rtl/matrix_mult.sv`)

`C = A × B` with A (M×K) and B (K×N) signed INT8 and C (M×N) signed INT32.
The default is 8×8×8, and M, K and N are parameters. This is the simplest
correct architecture: a single Phase 1 MAC does all the work.

```
          start
            |
 a_flat -->[ A regs, M x K ]-- A[i][k] --+
                                         +-->[ mac ]-- acc -->[ C regs, M x N ]--> c_flat
 b_flat -->[ B regs, K x N ]-- B[k][j] --+       ^                 ^
                                                 | en, clear       | write C[i][j]
           [ sequencer: k fastest, then j, then i ]+----------------+
```

- **Operands:** `start` copies A and B from flat, row-major buses into local
  registers.
- **Compute:** counters issue one multiply-accumulate per clock. The first
  term of each dot product uses the MAC's clear+en load, so consecutive dot
  products need no idle cycle.
- **Write-back:** each finished sum is written into C one cycle after its last
  term, while the next dot product is already running.
- **Timing:** latency is M·N·K + 1 = 513 cycles from the start edge to the
  one-cycle `done` pulse. Back-to-back runs start every M·N·K + 2 = 514
  cycles. The MAC is active for 512 of them.
- **Control:** a minimal counter sequencer. Phase 4 replaces it with a proper
  FSM and a valid/ready interface.

**Verification.** Two independent references check every result:

1. **Testbench (`tb/matrix_mult_tb.sv`):** compares all M·N elements against
   a 64-bit reference model, and checks that `done` arrives exactly at the
   documented latency.
2. **NumPy golden model (`python/golden_model.py`):** the testbench writes each
   multiply's A, B, and the C read back from the DUT to a results file, and
   `python/verify_results.py` recomputes C with NumPy and compares every
   element.

| Section   | What it covers                                                     |
|-----------|--------------------------------------------------------------------|
| Directed  | identity × B, A × identity, zeros, all 127, all −128, −128 × 127, a checkerboard of extremes, an index pattern with distinct elements |
| Protocol  | back-to-back starts, starts after idle cycles, start ignored while busy, reset in the middle of a run |
| Random    | 200 seeded matrix pairs, biased toward corner values               |
| Coverage  | positive, negative and zero elements; the largest (K·16384) and most negative (K·−16256) possible elements; each protocol case |

The golden model has its own self-test (`make golden`). It is checked against
plain Python integer arithmetic on 72 matrix pairs, plus hand-computed values.
It multiplies in int64 on purpose: NumPy's `int8 @ int8` wraps, and for the
all −128 case it returns 0 instead of 131,072.

Measured results (seed 1), with 0 errors on both Verilator 5.052 and Icarus
13.0, and every element matching NumPy:

| Shape (M×K×N) | Multiplies | Testbench checks | Elements vs. NumPy | Latency |
|---------------|------------|------------------|--------------------|---------|
| 8×8×8         | 210        | 13,866           | 13,440             | 513     |
| 3×16×5        | 210        | 3,576            | 3,150              | 241     |
| 1×16×8        | 210        | 2,106            | 1,680              | 129     |

For each shape, the Verilator and Icarus results files are byte-identical.
