# NN-Accelerator

An INT8 neural-network accelerator written in synthesizable SystemVerilog, fully
simulated and verified against a Python/NumPy golden model. No FPGA required.

The project is built incrementally; each phase must pass its tests before the
next one starts.

## Status

| Phase | Block                          | Status      |
|-------|--------------------------------|-------------|
| 1     | Signed INT8 MAC unit           | not started |
| 2     | ReLU                           | not started |
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
