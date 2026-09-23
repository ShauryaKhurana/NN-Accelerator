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
| 5     | NN layer `ReLU(X·W + B)`       | done — verified on Verilator and Icarus, and against NumPy |
| 6     | Two-layer network (16→32→10)   | done — verified on Verilator and Icarus, and against NumPy |
| 7     | Parallel MAC array             | done — verified on Verilator and Icarus, and against NumPy |
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
make parallel        # cycles per inference at NUM_MACS = 1, 4, 8, 16, 32
make mac-waves       # short run that writes waveforms/mac_tb.vcd (also relu-, ctrl-, matmul-, layer-, net-waves)
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

## Phase 5: fully connected layer (`rtl/nn_layer.sv`)

`Y = ReLU(X · W + B)` for one layer: X is a vector of INPUT_SIZE signed INT8
activations, W is INPUT_SIZE × OUTPUT_SIZE signed INT8 weights, B is one
signed INT32 bias per output, and Y is OUTPUT_SIZE signed INT32 values. The
default is 16 inputs and 8 outputs, and both are parameters.

The layer reuses the Phase 4 multiplier: a 1 × K by K × N multiply is exactly
X · W. It adds a bias, selected by a counter that follows the output beats,
and the Phase 2 ReLU. Both are combinational, so the layer keeps the
multiplier's schedule and adds no cycles.

- **Input stream:** X, then W row-major, one INT8 per beat (same protocol as
  the multiplier).
- **Bias:** one INT32 per output channel on a parallel port, held steady while
  the layer is busy. It is not on the stream because there is one per output,
  it is INT32 while the stream is INT8, and in a larger design it would sit in
  a small register file written once per layer.
- **Output stream:** Y, one INT32 per beat, `m_last` on the final output.
- **Numeric range:** any bias up to 2³¹ − 1 − INPUT_SIZE · 2¹⁴ in magnitude can
  never make the sum wrap (±2,147,221,503 for 16 inputs). The testbench checks
  the largest non-wrapping sum and one step past it, where the sum wraps
  negative and ReLU clamps the output to 0.

Full details, including the cycle-count table, are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Verification.** The testbench checks every output against a 64-bit reference
(dot product, bias with the documented wrap, then ReLU), and the same protocol
and cycle accounting as Phase 4: outputs hold steady while stalled, exact beat
counts, one-cycle `done`, exact busy cycles plus one per stall, and resets
during LOAD, COMPUTE and WRITEBACK. `python/verify_results.py layer` then
re-checks every logged inference against the NumPy golden model.

Directed cases cover zeros, bias only, all 127, sums that ReLU clamps to 0, a
bias that lifts a negative sum back up or pushes a positive sum below zero
(including the exact zero crossing), identity weights, and both sides of the
INT32 edge.

Measured results (seed 1), with 0 errors on both Verilator 5.052 and Icarus
13.0, and every logged output matching NumPy:

| Shape (inputs → outputs) | Inferences | Checks | Outputs vs. NumPy | Cycles per inference |
|--------------------------|------------|--------|-------------------|----------------------|
| 16 → 8                   | 215        | 4,122  | 1,712             | 282                  |
| 4 → 3                    | 215        | 1,962  | 642               | 33                   |
| 32 → 10                  | 215        | 4,986  | 2,140             | 684                  |

Each cycle count matches the formula in the architecture document. Loading the
weights is the bulk of it: 144 of the 282 cycles at 16 → 8.

## Phase 6: two-layer network (`rtl/nn_accelerator.sv`)

```
16 inputs → [ layer 1: 16×32, ReLU ] → requantize to INT8 → [ layer 2: 32×10 ] → 10 logits
```

Two `nn_layer` instances chained through a requantizer. Layer 2 is built with
`APPLY_RELU = 0`, because an output layer produces raw logits.

**Why a requantizer.** A layer accumulates in INT32, but the next layer's
multiplier takes INT8, so the activations have to come back down. This is the
usual fixed-point rescaling, with the scale restricted to a power of two:

```
x (INT32) → [ >>> SHIFT ] → [ saturate to INT8 ] → y (INT8)
```

- A constant shift costs no multiplier, just wiring plus a compare and a mux.
- The shift is arithmetic, so it rounds toward negative infinity, exactly like
  `>>` on a signed integer in Python and NumPy. Everything stays integer, so
  the hardware and the golden model agree exactly.
- Large activations saturate instead of wrapping.
- `SHIFT` is a parameter (default 8). With 16 INT8 inputs that keeps typical
  activations near 70 and saturates only the largest few percent.

**Routing.** One counter tracks the beats taken from the host: the first
16 + 512 feed layer 1, and the remaining 320 are W2. A second counter tracks
the activations handed to layer 2, and until all 32 have gone through,
`s_ready` stays low for W2, so the host waits. Layer 2 loads those activations
while layer 1 is still computing the later ones, so one inference costs less
than two separate layers.

**Interface.** Input stream: X, then W1, then W2, one INT8 per beat. Output
stream: 10 logits, one INT32 per beat, with `m_last` on the final one. Both
bias vectors sit on parallel ports, steady while busy.

| Phase of an inference | Cycles | 16 → 32 → 10 |
|---|---|---|
| Layer 1 loads X and W1 | INPUT·(1 + HIDDEN) | 528 |
| Layer 1 computes (+1 handover) | HIDDEN·(INPUT+1) + 1 | 545 |
| Layer 2 loads W2 | HIDDEN·OUTPUT | 320 |
| Layer 2 computes | OUTPUT·(HIDDEN+1) | 330 |
| IDLE + DONE | 2 | 2 |
| **Back to back** | | **1,725** |

**Verification.** The testbench checks every logit against a 64-bit reference
that repeats the whole pipeline, with the same protocol monitor and cycle
accounting as earlier phases, plus resets at four points (layer 1 load, layer
1 compute, W2 load, layer 2 compute). `python/verify_results.py network` then
recomputes each inference with NumPy, requantization included.

One directed test is worth calling out: with X = 0 and identity second-layer
weights, each logit *is* the requantized activation of its channel, so setting
biases to 0, one step below a step, exactly one step, 127 steps and beyond
reads the requantizer's 0, 1, 127 and saturation behavior straight off the
outputs.

Measured results (seed 1), with 0 errors on both Verilator 5.052 and Icarus
13.0, and every logit matching NumPy:

| Shape | SHIFT | Inferences | Checks | Logits vs. NumPy | Cycles per inference |
|-------|-------|------------|--------|------------------|----------------------|
| 16-32-10 | 8 | 112 | 2,611 | 1,120 | 1,725 |
| 16-32-10 | 0 | 112 | 2,611 | 1,120 | 1,725 |
| 4-3-2    | 8 | 111 | 793   | 222   | 48    |

Of the 1,725 cycles, 848 go to loading weights and 874 to computing. Phases 7
and 8 target exactly that.

## Phase 7: parallel MACs (`NUM_MACS`)

`NUM_MACS` sets how many output columns are computed at once. Every MAC sees
the same activation; MAC *p* reads its own weight `B[k][j_base + p]` and keeps
its own accumulator. A group of that many columns takes K cycles whatever its
width, and then streams out one column per cycle:

```
compute + write-back cycles = M · ceil(N / NUM_MACS) · K + M · N
```

When N is not a multiple of `NUM_MACS` the last group is short; the spare MACs
repeat the last column so their addresses stay in range, and their results are
never streamed out. `NUM_MACS = 1` is the original schedule.

**Measured cycles** (`make parallel`). Simulated cycle counts only: no clock
frequency is implied, and none of this is an FPGA measurement.

| NUM_MACS | Network (16-32-10) | Speedup | Layer 16→32 | Layer compute only |
|----------|--------------------|---------|-------------|--------------------|
| 1        | 1,725              | 1.00×   | 1,074       | 544                |
| 4        | 1,117              | 1.54×   | 690         | 160                |
| 8        | 1,021              | 1.69×   | 626         | 96                 |
| 16       | 957                | 1.80×   | 594         | 64                 |
| 32       | 941                | 1.83×   | 578         | 48                 |

The compute phase scales nearly linearly — the 16→32 layer's compute falls
from 544 cycles to 48, which is 11.3× with 32 MACs — but the end-to-end figure
does not, because weight loading is unchanged at one byte per cycle. For the
network, 848 of the 1,725 baseline cycles are weight beats, and at 32 MACs
those same 848 cycles are 90% of the remaining 941. Past about 8 MACs, more
multipliers buy very little; the bottleneck is the weight port, not the
arithmetic.

**Verification.** Every existing testbench takes `NUM_MACS` and checks the
cycle formula exactly for that width, and the NumPy checks run at each one, so
the parallel configurations are proven to compute identical results, not just
similar ones. The regression runs the matrix multiply at 1 and 4 MACs (square
and non-square), the controller at 1 and 4, the layer at 1 and 8, and the
network at 1, 4 and 8.
