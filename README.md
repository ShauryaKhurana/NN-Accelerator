# NN-Accelerator

An INT8 neural-network accelerator written in synthesizable SystemVerilog, fully
simulated and verified against a Python/NumPy golden model. No FPGA required.

The project is built incrementally; each phase must pass its tests before the
next one starts.

## Status

| Phase | Block                          | Status      |
|-------|--------------------------------|-------------|
| 1     | Signed INT8 MAC unit           | done — verified on Verilator and Icarus |
| 2     | ReLU                           | done — verified on Verilator and Icarus |
| 3     | 8×8 matrix multiply            | not started |
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
python/     golden model, vector generation, result checking
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

Python dependencies (needed from Phase 3). Homebrew's Python refuses global
`pip install` (PEP 668), so use a project virtual environment:

```sh
python3 -m venv .venv
.venv/bin/pip install numpy
```

Check what is installed with `make check-tools`.

## Running the tests

```sh
make test            # Verilator: RTL lint + all testbenches (primary)
make test-iverilog   # Icarus Verilog: the same testbenches (cross-check)
make mac SEED=1234   # one testbench with a different random seed
make mac-waves       # short run that writes waveforms/mac_tb.vcd (also: relu-waves)
make help            # list every target
```

Every testbench prints a single `TEST PASSED` / `TEST FAILED` line. A failure
exits nonzero, so `make` stops. Logs go to `sim/<run>.<simulator>.log`.

To view a waveform: `surfer waveforms/mac_tb.vcd` (GTKWave also reads VCD if
you have it). Waveform runs record only a testbench's directed sections, which
keeps the files around 20 KB. Add `+dumpall` to record everything.

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
