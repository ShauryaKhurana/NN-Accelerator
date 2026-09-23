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
| 4     | Control FSM + handshaking      | done — verified on Verilator and Icarus, and against NumPy |
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
make mac-waves       # short run that writes waveforms/mac_tb.vcd (also relu-, ctrl-, matmul-waves)
make help            # list every target
```

Every testbench prints a single `TEST PASSED` / `TEST FAILED` line. A failure
exits nonzero, so `make` stops. Logs go to `sim/<run>.<simulator>.log`.
From a clean tree, `make test` takes about 30 s and `make test-iverilog`
about 13 s (Apple M-series).

**If a Verilator build occasionally stalls for minutes on macOS:** every
Verilator build produces a new, ad-hoc-signed executable. Gatekeeper scans it
on first launch, and the scan includes an online check. On some networks that
check can hang for 1–15 minutes. While it waits, other process launches
(including the next build's compiler) queue behind it, and all tests still
pass afterwards. To exempt your terminal, add it (Terminal, or Visual Studio
Code for its integrated terminal) under System Settings → Privacy & Security →
Developer Tools. Icarus is unaffected, because it creates no new executables.

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

## Phase 3: matrix multiply and the NumPy golden model

`C = A × B`, where A (M×K) and B (K×N) are signed INT8 and C (M×N) is signed
INT32. The default is 8×8×8, and M, K and N are parameters. This is the
simplest correct architecture: a single Phase 1 MAC computes the M·N dot
products one after another, one multiply-accumulate per clock. The first
term of each dot product uses the MAC's clear+en load, so consecutive dot
products need no idle cycle.

Phase 3's first version used wide parallel buses and a counter sequencer,
with a latency of 513 cycles for 8×8×8 (PR #3). Phase 4 replaced its
interface and control; the datapath schedule is unchanged.

**NumPy golden model.** `python/golden_model.py` defines the arithmetic the
RTL must match. The matrix testbench writes each job's A, B and the C it
received from the DUT to a results file. `python/verify_results.py` then
recomputes C with NumPy and compares every element, so every result is
checked by two independent references.

The golden model has its own self-test (`make golden`). It is checked against
plain Python integer arithmetic on 72 matrix pairs, plus hand-computed values.
It multiplies in int64 on purpose: NumPy's `int8 @ int8` wraps, and for the
all −128 case it returns 0 instead of 131,072.

## Phase 4: controller FSM and streaming interface

The matrix multiplier is now a streaming block with valid/ready handshakes,
sequenced by a separate controller (`rtl/matmul_ctrl.sv`).

- **Input stream** (`s_valid`, `s_ready`, `s_data`): A, then B, row-major, one
  INT8 element per beat.
- **Output stream** (`m_valid`, `m_ready`, `m_data`, `m_last`): C, row-major,
  one INT32 element per beat. `m_last` marks the final element.
- **FSM:** IDLE → LOAD → (COMPUTE → WRITEBACK, once per element of C) → DONE
  → IDLE.
- **No combinational paths:** `s_ready` and `m_valid` depend only on the
  registered state.
- **No C buffer:** each element of C streams straight from the MAC's
  accumulator.

The state diagram, transition table, output decode, protocol and timing
diagram are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

| Phase of a job      | Cycles (general) | 8×8×8 |
|---------------------|------------------|-------|
| IDLE, seeing `s_valid` | 1             | 1     |
| LOAD                | M·K + K·N        | 128   |
| COMPUTE + WRITEBACK | M·N·(K + 1)      | 576   |
| DONE                | 1                | 1     |
| **Back to back**    | **M·K + K·N + M·N·(K+1) + 2** | **706** |

Each stall cycle (`s_valid` low in LOAD, or `m_ready` low in WRITEBACK) adds
exactly one cycle. With no stalls, the MAC is busy for 512 of the 706
cycles.

**Verification.** There are two testbenches, and the NumPy check still runs
on every job:

1. **Controller testbench (`tb/matmul_ctrl_tb.sv`):** tests the FSM on its
   own. Each cycle it checks every output against a reference model written
   from the transition table. After every clock edge it checks the state and
   counters. It also checks busy-cycle accounting per job under random
   stalls, resets from every busy state, and requires every legal transition
   to be covered.
2. **Matrix testbench (`tb/matrix_mult_tb.sv`):** streams A and B in and C
   out, with random input gaps and output backpressure. It checks every
   element of C and the position of `m_last`. A monitor checks the protocol
   on every clock edge: the output holds while stalled, each job has exactly
   the right number of input and output beats, and `done` is a single-cycle
   pulse. It also checks exact cycle accounting.
   - Directed matrices run back to back, and each must finish exactly 706
     cycles after the previous one.
   - A source queues the next job while the current one computes; the DUT
     must not accept it early.
   - Resets during LOAD, COMPUTE and WRITEBACK must each be followed by an
     exact clean job.

Measured results (seed 1), with 0 errors on both Verilator 5.052 and Icarus
13.0:

| Controller testbench | Jobs | Cycles | Checks  | Busy cycles per job |
|----------------------|------|--------|---------|---------------------|
| 8×8×8                | 48   | 57,665 | 519,007 | 705                 |
| 2×3×2                | 408  | 31,742 | 286,060 | 29                  |
| 1×1×1                | 408  | 7,984  | 72,238  | 5                   |

| Matrix testbench | Jobs | Checks | Elements vs. NumPy | Back-to-back period |
|------------------|------|--------|--------------------|---------------------|
| 8×8×8            | 216  | 28,316 | 13,824             | 706                 |
| 3×16×5           | 216  | 7,148  | 3,240              | 385                 |
| 1×16×8           | 216  | 4,124  | 1,728              | 282                 |

Every element matches NumPy, and for each shape the Verilator and Icarus
results files are byte-identical.
