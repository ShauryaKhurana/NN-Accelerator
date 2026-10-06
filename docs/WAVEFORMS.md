# Waveforms

Every testbench can write a waveform. Nothing is dumped unless you ask for it,
so the normal test runs stay fast.

## Generating one

```sh
make net-waves                  # waveforms/nn_accelerator_tb.vcd
make net-waves WAVE_FORMAT=fst  # waveforms/nn_accelerator_tb.fst
```

The same for `mac-waves`, `relu-waves`, `ctrl-waves`, `matmul-waves` and
`layer-waves`. The Python regression can trace too:

```sh
.venv/bin/python python/run_regression.py --waves fst --max-cases 3
```

### VCD or FST

| | VCD | FST |
|---|---|---|
| Format | text | compressed binary |
| Network run above | 2.6 MB | 147 KB |
| Regression, 3 cases | 1.1 MB | 35 KB |
| Written by | Verilator, Icarus | Verilator only |

FST is about 18x smaller here and worth using for anything long. It is a
compile-time choice in Verilator, so switching `WAVE_FORMAT` rebuilds.

Two things that will waste your time if you do not know them:

- **Icarus Verilog only writes VCD.** Give it a `.fst` filename and it will
  happily write VCD content into it, print `VCD info: dumpfile ... opened`,
  and produce a file no viewer will open. The `fst` targets here are
  Verilator-only for that reason.
- **`$dumpoff` is ignored** by Verilator, so a testbench cannot pause its own
  dump. The only way to keep a trace small is to run less: the waveform
  targets run section 1 only, and the regression takes `--max-cases`.

### FST needs lz4

Verilator's FST writer includes `<lz4.h>`. If it is missing:

```
fatal error: 'lz4.h' file not found
```

```sh
brew install lz4
```

Homebrew does not put its headers on Apple clang's default search path, so the
Makefile passes `-I$(brew --prefix lz4)/include` itself. Override `LZ4_PREFIX`
if lz4 lives elsewhere.

## Opening one

**Surfer** (installed here, version 0.7.0):

```sh
surfer waveforms/nn_accelerator_tb.fst
```

**GTKWave** — the viewer the project brief names. It is *not* installed here,
so the commands below are from its documentation rather than something this
repo has exercised:

```sh
brew install --cask gtkwave
gtkwave waveforms/nn_accelerator_tb.fst
```

In GTKWave: pick a scope in the top-left *SST* pane, select signals in the
pane below it, and press **Insert** (or *Append*) to add them to the wave
window. **File > Write Save File** stores the signal selection as a `.gtkw`
you can reopen with `gtkwave -a saved.gtkw dump.fst`. Installing GTKWave also
brings `vcd2fst` and `fst2vcd`, which convert an Icarus VCD after the fact.

Both viewers read VCD and FST.

## What to look at

Signal names below are as they appear in `waveforms/nn_accelerator_tb.vcd`,
checked against the file rather than written from memory. The top scope is the
testbench; `dut` is the accelerator.

### The interface

| Want | Signal |
|---|---|
| clock | `nn_accelerator_tb.clk` |
| reset | `nn_accelerator_tb.rst` |
| cycle counter | `nn_accelerator_tb.cycle` |
| job kind | `nn_accelerator_tb.load_weights` |
| input valid / ready | `s_valid`, `s_ready` |
| input data | `s_data` |
| output valid / ready | `m_valid`, `m_ready` |
| output data | `m_data`, `m_last` |
| busy / done | `busy`, `done` |

A beat transfers on a rising edge where valid and ready are both high. That is
the first thing to check when something stalls: `s_valid` high with `s_ready`
low means the source is waiting for the DUT, and the reverse means the DUT is
starved.

### Inside a layer

Replace `u_layer1` with `u_layer2` for the output layer.

| Want | Signal |
|---|---|
| FSM state | `dut.u_layer1.u_matmul.u_ctrl.state` |
| load address | `dut.u_layer1.u_matmul.u_ctrl.ld_idx`, `ld_en` |
| MAC enables | `dut.u_layer1.u_matmul.u_ctrl.mac_en`, `mac_clear` |
| operand indices | `dut.u_layer1.u_matmul.u_ctrl.i`, `j_base`, `k` |
| write-back select | `dut.u_layer1.u_matmul.u_ctrl.wb_sel` |
| accumulator | `dut.u_layer1.u_matmul.g_mac[0].u_mac.acc` |
| resident weights | `dut.u_layer1.u_matmul.op_mem` |
| layer output | `dut.u_layer1.dot` (before bias/ReLU) |
| requantized activation | `dut.act` (INT8 into layer 2) |

`state` is the enum from `rtl/matmul_pkg.sv`, dumped as a 3-bit number:

| Value | State |
|---|---|
| 0 | IDLE |
| 1 | LOAD |
| 2 | COMPUTE |
| 3 | WRITEBACK |
| 4 | DONE |

`op_mem` is one array per layer holding the activations then the weights. It
only appears in the dump because the Makefile raises Verilator's
`--trace-max-array` and `--trace-max-width`; under the defaults (32 entries,
256 bits) Verilator leaves the whole array out silently rather than truncating
it, which is easy to mistake for the signal not existing.

### Reading a job

With weights resident there are two kinds of job, and `load_weights` tells
them apart at the first beat.

A **weight load** is `state` IDLE -> LOAD for K*N beats -> DONE, with
`m_valid` never rising. Watch `ld_idx` climb through the weight region of
`op_mem`.

An **inference** is IDLE -> LOAD for the activations -> then COMPUTE and
WRITEBACK alternating: `mac_en` high for K cycles with `mac_clear` on the
first of them, `acc` accumulating, then one cycle of `m_valid` per output
column. `i`, `j_base` and `k` say which element is being computed. For the
network, layer 2's LOAD overlaps layer 1's WRITEBACK — put both layers'
`state` signals next to each other and the handoff is the clearest thing on
the screen.

Cycle counts for each phase are in [ARCHITECTURE.md](ARCHITECTURE.md).
