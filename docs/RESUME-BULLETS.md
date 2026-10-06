# Resume bullet options

Every number below is a measurement from this repository, reproducible with
`make test`, `make regression` and `make perf`. Nothing here claims an FPGA
result, a clock frequency, an area figure or a speedup against real hardware,
because none of those were measured — the project is simulation only.

Pick two or three depending on what the role emphasises.

---

### Option 1 — design and microarchitecture

> Designed a parameterized INT8 neural-network accelerator in synthesizable
> SystemVerilog (~950 lines, 8 modules) implementing a two-layer fully
> connected network with a configurable parallel MAC array, AXI-Stream-style
> valid/ready interfaces, and separated datapath and control. Making the
> weights resident on-chip instead of re-streaming them per inference raised
> the ceiling on parallel-MAC scaling from 1.8x to 11.3x, cutting a 16-32-10
> inference from 1,725 to 76 simulated cycles.

### Option 2 — verification

> Built a dual-flow verification environment for an INT8 accelerator: nine
> self-checking SystemVerilog testbenches that model cycle-exact timing under
> randomized stalls, plus a Python/NumPy-driven regression that generates test
> vectors, drives the RTL and recomputes every result. Cross-checked on two
> simulators (Verilator and Icarus) with byte-identical results, verifying
> ~128,000 values against the golden model with zero mismatches; validated the
> suite itself by injecting 68 deliberate RTL bugs, of which it caught 66.

### Option 3 — performance analysis

> Instrumented an INT8 accelerator's RTL testbenches to report cycle-exact
> per-phase timing, then automated a performance sweep in Python across MAC
> array widths of 1 to 32. Measured throughput rising 11.3x while MAC
> utilization fell from 96.7% to 34.2%, identifying the fixed-cost input load,
> inter-layer handoff and single-wide output port as the scaling limit, and
> showing the design reaches the slower layer's own pipeline period at 32 MACs.

---

## Supporting numbers, if asked

| Claim | Where it comes from |
|---|---|
| ~950 lines of RTL, 8 modules | `wc -l rtl/*.sv` |
| 1,725 -> 76 cycles per inference | `make perf`; 1,725 is the pre-weight-reuse figure in [BUILD-LOG.md](BUILD-LOG.md), 76 is the current streamed interval at 32 MACs |
| 1.8x -> 11.3x scaling ceiling | 1.83x was the best speedup when weights streamed every inference; 11.32x is the current measured speedup at 32 MACs |
| 96.7% -> 34.2% MAC utilization | 832 MAC operations per inference against the active array's width x the measured interval |
| ~128,000 values vs NumPy | 53,531 from the testbench flow plus 74,790 from the regression flow |
| two simulators, byte-identical | Verilator 5.052 and Icarus Verilog 13.0; the results files match per shape |
| 66 of 68 injected bugs caught | per-phase table in [VERIFICATION.md](VERIFICATION.md); both survivors explained there |
| 9 testbenches | 7 self-checking + 2 vector-driven, `ls tb/` |

## What not to claim

- No clock frequency, LUT, DSP, BRAM or power numbers — nothing was synthesized.
- No speedup against a CPU, GPU or any real device. The speedups are ratios of
  simulated cycle counts within this design.
- "Throughput" means inferences per simulated cycle, not per second.
- The network is 16-32-10 and fully connected only. It is a working accelerator
  for a small MLP, not a production inference engine.
