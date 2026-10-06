# Architecture

This document describes the hardware as built so far. It will grow as later
phases add pipelining.

## Blocks

| Block          | File                  | Role                                                        |
|----------------|-----------------------|-------------------------------------------------------------|
| MAC            | `rtl/mac.sv`          | Signed INT8 × INT8 → INT32 multiply-accumulate, 1 per clock |
| ReLU           | `rtl/relu.sv`         | `y = x < 0 ? 0 : x` on INT32, combinational                 |
| Matrix multiply| `rtl/matrix_mult.sv`  | Streaming C = A × B datapath: operand memory + NUM_MACS MACs|
| Controller     | `rtl/matmul_ctrl.sv`  | FSM that sequences a matrix-multiply job                    |
| Layer          | `rtl/nn_layer.sv`     | `Y = ReLU(X · W + B)`: the multiplier plus bias and ReLU    |
| Requantizer    | `rtl/requant.sv`      | INT32 activation → INT8: shift, then saturate               |
| Network        | `rtl/nn_accelerator.sv` | Two layers chained through the requantizer                |
| Shared types   | `rtl/matmul_pkg.sv`   | Controller state encoding                                   |

## Matrix multiplier

`C = A × B`, where A (M×K) and B (K×N) are signed INT8 and C (M×N) is
signed INT32. The default is M = K = N = 8, and all three are parameters.

```
                      +--------------------------------+
 s_valid, s_ready <-->|          matmul_ctrl           |<--> m_valid, m_ready, m_last
                      | IDLE LOAD COMPUTE WRITEBACK .. |---> busy, done
                      +--------------------------------+
                        | ld_en   | i, j_base, k, wb_sel | mac_en, mac_clear
                        v ld_idx   v                      v
 s_data --------->[ operand memory ]-- A[i][k] ------> broadcast to every MAC
 (INT8)           [ A, then B:     ]-- B[k][j_base+p] -> MAC p
                  [ M*K + K*N B    ]   NUM_MACS accumulators --mux--> m_data
```

- **Operand memory.** A and B are stored back to back in one memory,
  M·K + K·N bytes (128 for 8×8×8). It has one write port, used while
  loading, and two read ports: A[i][k] at address `i*K + k`, and B[k][j] at
  address `M*K + k*N + j`.
- **MACs.** Each element of C is a K-term dot product, computed in K
  consecutive cycles. The first term uses the MAC's clear+en load, so a new
  sum starts without an idle cycle. With `NUM_MACS > 1`, that many output
  columns are computed at once; see *Parallel MACs* below.
- **No C buffer.** The MACs' accumulator registers are the output data
  registers. While an element waits to be accepted, the MACs are idle, and
  their accumulators hold their values steady. (Phase 3's first version kept a
  2,048-bit C register file; the streaming design does not need it.)
- **Reset.** Only control state is reset. The operand memory is data and is
  always written before it is read.

### Stream interface

| Signal    | Dir | Width | Meaning |
|-----------|-----|-------|---------|
| `s_valid` | in  | 1     | the source has an input element on `s_data` |
| `s_ready` | out | 1     | the DUT will accept it (high only in LOAD) |
| `s_data`  | in  | 8     | signed INT8 element |
| `m_valid` | out | 1     | an element of C is on `m_data` (high only in WRITEBACK) |
| `m_ready` | in  | 1     | the sink will accept it |
| `m_data`  | out | 32    | signed INT32 element of C |
| `m_last`  | out | 1     | this is the final element of C |
| `busy`    | out | 1     | a job is in progress (any state but IDLE) |
| `done`    | out | 1     | one-cycle pulse after the final output beat |

- **Transfers.** A beat transfers on a rising edge where valid and ready are
  both high.
- **Input order.** M·K elements of A, then K·N elements of B, each matrix
  row-major, one element per beat.
- **Output order.** M·N elements of C, row-major, one element per beat, with
  `m_last` on the final one.
- **Output stability.** Once `m_valid` is high, it stays high, with `m_data`
  and `m_last` unchanged, until the beat is accepted.
- **No combinational paths.** `s_ready` and `m_valid` depend only on the
  registered state, never directly on `s_valid` or `m_ready`. Chaining the
  multiplier with other valid/ready blocks therefore cannot create a
  combinational loop.

### Controller FSM

```
          s_valid        last A beat (inference)            k == K-1
+------+ -------> +------+ ---------------------> +---------+ ------> +-----------+
| IDLE |          | LOAD |                        | COMPUTE |         | WRITEBACK |
+------+          +------+                        +---------+ <------ +-----------+
    ^                 |                                      m_ready,        |
    |                 | last B beat (weight load)            more left       | m_ready,
    |                 v                                                      | last element
    |             +------+                                                   |
    +-------------| DONE |<--------------------------------------------------+
                  +------+
```

A job is a weight load or an inference. `load_weights`, sampled with the first
beat, picks which: a weight load takes K·N beats into the B region and goes
straight to DONE, while an inference takes M·K beats of A and goes on to
COMPUTE. The weights stay put between jobs.

| State     | Leaves when                                   | Goes to   | Counters |
|-----------|-----------------------------------------------|-----------|----------|
| IDLE      | `s_valid`                                     | LOAD      | all cleared; the job kind is latched from `load_weights`, and it sets the first load address |
| LOAD      | the last of K·N beats (weight load)           | DONE      | `ld_idx` + 1 per accepted beat |
| LOAD      | the last of M·K beats (inference)             | COMPUTE   | `ld_idx` + 1 per accepted beat |
| COMPUTE   | `k == K-1` (after K cycles)                   | WRITEBACK | `k` + 1, wrapping to 0 |
| WRITEBACK | `m_ready`, and the group has more columns     | WRITEBACK | `wb_sel` + 1 |
| WRITEBACK | `m_ready`, group finished, more elements left | COMPUTE   | next group (`j_base` + NUM_MACS), then next row |
| WRITEBACK | `m_ready`, and this was the last element      | DONE      | |
| DONE      | always (one cycle)                            | IDLE      | |

Each state holds until its condition is true. Reset returns to IDLE from any
state. The encoding is fixed in `rtl/matmul_pkg.sv` (IDLE = 0, LOAD = 1,
COMPUTE = 2, WRITEBACK = 3, DONE = 4), so the state is easy to read as a
number in a waveform.

All outputs are decoded from the registered state and counters:

| Output      | High when                              |
|-------------|----------------------------------------|
| `s_ready`   | LOAD                                   |
| `ld_en`     | LOAD and `s_valid` (operand write)     |
| `mac_en`    | COMPUTE                                |
| `mac_clear` | COMPUTE and `k == 0` (first term)      |
| `m_valid`   | WRITEBACK                              |
| `m_last`    | WRITEBACK and the element is C[M-1][N-1] |
| `busy`      | not IDLE                               |
| `done`      | DONE                                   |

IDLE deliberately does not accept data, so LOAD is the only state that
writes operands. This costs one cycle per job.

### Timing

A weight load, with no stalls (8×8×8). It never leaves LOAD for COMPUTE:

```
cycle          0      1      2    ...    64     65     66
state         IDLE   LOAD   LOAD  ...   LOAD   DONE   IDLE
load_weights   1      1      1           1
s_valid        1      1      1           1      0
s_ready        0      1      1           1      0
beat                  B00    B01         B77
done                                             1
```

The start of an inference against those weights:

```
cycle          0      1      2    ...    64     65   ...    72     73     74   ...
state         IDLE   LOAD   LOAD  ...   LOAD  COMPUTE ... COMPUTE  WB   COMPUTE ...
load_weights   0      0      0           0
s_valid        1      1      1           1      0
s_ready        0      1      1           1      0
beat                  A00    A01         A77
k                                               0     ...    7            0
mac_clear                                       1                         1
m_valid                                                              1
m_data                                                             C00
```

The end of the inference: C[7][7] is computed in cycles 632–639. Its
write-back beat is cycle 640, and DONE is cycle 641. In cycle 642 the next
inference's IDLE cycle can start.

Weights are loaded once and reused, so there are two kinds of job. With
`NUM_MACS = 1`:

| Job           | Phase                  | Cycles (general)          | 8×8×8 |
|---------------|------------------------|---------------------------|-------|
| Weight load   | IDLE, seeing `s_valid` | 1                         | 1     |
|               | LOAD (B)               | K·N                       | 64    |
|               | DONE                   | 1                         | 1     |
|               | **back to back**       | **K·N + 2**               | **66** |
| Inference     | IDLE, seeing `s_valid` | 1                         | 1     |
|               | LOAD (A)               | M·K                       | 64    |
|               | COMPUTE + WRITEBACK    | M·ceil(N/NUM_MACS)·K + M·N | 576  |
|               | DONE                   | 1                         | 1     |
|               | **back to back**       | **M·K + M·ceil(N/NUM_MACS)·K + M·N + 2** | **642** |

At eight MACs an 8×8×8 inference is 194 cycles; the weight load is unchanged
at 66.

- **Busy time.** `busy` is high for all but the IDLE cycle: 65 cycles for a
  weight load, 641 for an 8×8×8 inference.
- **Stalls.** Every cycle with `s_valid` low during LOAD, or `m_ready` low
  during WRITEBACK, adds exactly one cycle. Both testbenches check this
  cycle for cycle, under random stalls.
- **Measured inference periods.** Back to back with no stalls: 642 cycles for
  8×8×8, 305 for 3×16×5, and 154 for 1×16×8. Each matches the formula.
- **Where the time goes (8×8×8).** Of the 642 inference cycles, the MAC is
  busy for M·N·K = 512 (79.8%). The rest is loading A (64 cycles),
  write-back beats (64), and IDLE plus DONE (2).

## Fully connected layer

`Y = ReLU(X · W + B)` for one layer: X is a vector of INPUT_SIZE signed INT8
activations, W is INPUT_SIZE × OUTPUT_SIZE signed INT8 weights, B is one
signed INT32 bias per output, and Y is OUTPUT_SIZE signed INT32 values. The
default is 16 inputs and 8 outputs.

```
 s_data -->[ matrix_mult (M=1, K=INPUT_SIZE, N=OUTPUT_SIZE) ]--+
 (X then W, INT8)        X . W[:,j], INT32                     |
                                                               v
                        B[j] ------------------------------>[ + ]
                     (INT32, held steady)                      |
                                                               v
                                                           [ relu ] --> m_data (INT32)
```

- **Reuse.** A 1 × K by K × N multiply is exactly X · W, so the layer
  instantiates the matrix multiplier with M = 1 and adds two things: the bias
  and the ReLU.
- **Bias select.** The multiplier emits one beat per output channel, in
  order, so a counter that follows the output handshakes picks the matching
  bias. It returns to channel 0 on the beat carrying `m_last`.
- **No extra cycles.** The bias add and ReLU are combinational, so the layer
  keeps the multiplier's schedule exactly. While an output waits to be
  accepted, the MAC holds its accumulator, so the offered value is stable.
- **Bias interface.** The bias is a parallel input, not part of the stream:
  there is one per output channel, it is INT32 while the stream is INT8, and
  in a larger design it would live in a small register file written once per
  layer. It must hold steady while the layer is busy.

### Numeric range

A dot product of K INT8 pairs is at most K · 2¹⁴ in magnitude, which leaves
plenty of INT32 headroom. Adding the bias wraps modulo 2³² like any
two's-complement add, so any bias with magnitude up to 2³¹ − 1 − K · 2¹⁴
can never wrap; for K = 16 that is ±2,147,221,503. The testbench checks the
largest non-wrapping sum (which gives exactly 2,147,483,647) and one step
past it, where the sum wraps negative and ReLU clamps the output to 0.
ReLU is applied after the bias, so Y is never negative.

### Cycles per inference

| Job         | Phase                  | Cycles (general)                    | 16 → 8 |
|-------------|------------------------|-------------------------------------|--------|
| Weight load | IDLE + LOAD (W) + DONE | INPUT_SIZE·OUTPUT_SIZE + 2          | 130    |
| Inference   | IDLE, seeing `s_valid` | 1                                   | 1      |
|             | LOAD (X)               | INPUT_SIZE                          | 16     |
|             | COMPUTE + WRITEBACK    | OUTPUT_SIZE·(INPUT_SIZE + 1)        | 136    |
|             | DONE                   | 1                                   | 1      |
|             | **back to back**       | **the sum of the above**            | **154** |

The bias is a parallel INT32 port, not part of the stream, so it can change
from one inference to the next without reloading anything.

Measured back-to-back inference periods: 154 cycles at 16 → 8, 21 at 4 → 3,
364 at 32 → 10, and 562 at 16 → 32, each matching the formula. Weight loading
used to dominate this figure — 144 of 282 cycles — and has moved out of the
per-inference path entirely.

## Requantizer

Between two INT8 layers the activations have to come back down to INT8: a
layer accumulates in INT32, but the next layer's multiplier takes INT8.

```
  x (INT32) --->[ >>> SHIFT ]--->[ saturate to OUT_WIDTH ]---> y (INT8)
                 arithmetic          clamp, do not wrap
```

- **Power-of-two scale.** Restricting the rescaling factor to a power of two
  makes it a constant shift, which is wiring, plus two comparisons and a mux.
  The general version multiplies by a fixed-point scale first.
- **Rounding.** The shift is arithmetic, so it rounds toward negative
  infinity, the same as `>>` on a signed integer in Python or NumPy. The
  hidden activations arrive after ReLU, so they are never negative, but the
  module is tested with negatives too.
- **Saturation, not wrapping.** A large activation becomes the largest
  representable value instead of changing sign.
- **Choosing SHIFT.** With K INT8 inputs, sums reach about K · 2¹⁴. The
  default 8 keeps typical activations of a 16-input layer around 70 and
  saturates only the largest few percent. It is a parameter, so it can be
  swept.

## Two-layer network

```
  s_data ---+--->[ nn_layer 1: INPUT_SIZE -> HIDDEN_SIZE, ReLU ]
  (INT8)    |            |
  X, W1, W2 |            | activations (INT32, non-negative)
            |            v
            |      [ requant: >>> SHIFT, saturate ]
            |            | INT8
            |            v
            +--->[ nn_layer 2: HIDDEN_SIZE -> OUTPUT_SIZE, no ReLU ]---> m_data
              W2                                                         (INT32 logits)
```

The default shape is 16 → 32 → 10. Layer 2 has `APPLY_RELU = 0`, because the
output layer produces raw logits.

- **Routing.** One counter tracks how many beats have been taken from the
  host. The first INPUT_SIZE + INPUT_SIZE·HIDDEN_SIZE of them feed layer 1. A
  second counter tracks the activations handed to layer 2. Until all
  HIDDEN_SIZE of them have gone through, `s_ready` stays low for the W2 beats,
  so the host simply waits.
- **Overlap.** Layer 2 loads its activations while layer 1 is still computing
  the later ones, so an inference costs less than two separate layers.

### Cycles per inference

A weight load streams W1 then W2; the host holds `load_weights` high for the
whole job, and an internal beat counter splits the stream between the layers.
An inference streams only X.

| Job         | Phase                                           | Cycles                          | 16 → 32 → 10 |
|-------------|-------------------------------------------------|---------------------------------|--------------|
| Weight load | layer 1 takes W1 (+ its DONE)                   | INPUT_SIZE·HIDDEN_SIZE + 1      | 513          |
|             | layer 2 takes W2 (+ its DONE)                   | HIDDEN_SIZE·OUTPUT_SIZE + 1     | 321          |
|             | **busy**                                        | **the sum of the above**        | **834**      |
| Inference   | IDLE, seeing the first beat                     | 1                               | 1            |
|             | layer 1 loads X                                 | INPUT_SIZE                      | 16           |
|             | layer 1 computes (+1 while layer 2 leaves IDLE) | HIDDEN_SIZE·(INPUT_SIZE+1) + 1  | 545          |
|             | layer 2 computes                                | OUTPUT_SIZE·(HIDDEN_SIZE+1)     | 330          |
|             | DONE                                            | 1                               | 1            |
|             | **back to back**                                | **the sum of the above**        | **893**      |

Measured: 892 busy cycles and 893 between back-to-back inferences at
16 → 32 → 10, and 30 at 4 → 3 → 2, both matching the formula. The weight load
is paid once: 834 cycles, after which each inference costs 893.

## Parallel MACs

`NUM_MACS` sets how many output columns are computed at once. Every MAC sees
the same activation `A[i][k]`; MAC *p* reads its own weight `B[k][j_base + p]`
and keeps its own accumulator.

```
                      A[i][k]  (broadcast)
                         |
        +----------------+----------------+
        v                v                v
   [ mac 0 ]        [ mac 1 ]   ...  [ mac P-1 ]
   B[k][j]          B[k][j+1]        B[k][j+P-1]
        |                |                |
        +--------- mux (wb_sel) ----------+---> m_data, one column per cycle
```

- A group of `NUM_MACS` columns takes K cycles whatever its width, then
  streams out one column per cycle.
- When N is not a multiple of `NUM_MACS`, the last group is short. The spare
  MACs repeat the last column so their addresses stay in range, and their
  results are never streamed out.
- `NUM_MACS = 1` is the original single-MAC schedule.

```
compute + write-back cycles = M · ceil(N / NUM_MACS) · K + M · N
```

The first term falls with `NUM_MACS`; the second, one beat per output element,
does not. Nor does weight loading, which is one byte per cycle.

### Measured cycles

`make parallel` runs the 16-32-10 network at each width and prints this table.
These are simulated cycle counts. No clock frequency is implied, and nothing
here is an FPGA measurement.

| NUM_MACS | Network cycles/inference | Speedup | Weight load (once) |
|----------|--------------------------|---------|--------------------|
| 1        | 893                      | 1.00×   | 834                |
| 4        | 285                      | 3.13×   | 834                |
| 8        | 189                      | 4.72×   | 834                |
| 16       | 125                      | 7.14×   | 834                |
| 32       | 109                      | 8.19×   | 834                |

The weight load is a fixed 834 cycles at every width: it is one byte per cycle
through the same port, and more MACs do not widen it.

These numbers are from Phase 8, with weights resident. Before that, every
inference reloaded them, and the same sweep gave 1,725 / 1,117 / 1,021 / 957 /
941 cycles — a 1.83× ceiling, because 848 of the 1,725 baseline cycles were
weight beats and no number of MACs could shrink them. That was Amdahl's law
with concrete numbers. Taking the weights out of the per-inference path raises
the ceiling to 8.19×, and the two changes together take the 16-32-10 inference
from 1,725 cycles to 109.

## Pipelining, latency and throughput

### Stages

A value passes through four pieces of hardware on its way from the input
stream to the output stream. Only two of them cost cycles:

```
  s_data ──▶ [1] operand    ──▶ [2] MAC     ──▶ [3] bias ──▶ [4] requantize ──▶ m_data /
             memory             array           + ReLU        (INT32→INT8)      next layer
             1 beat/cycle       K cycles        same cycle    same cycle
             registered         registered      combinational combinational
```

Stages 3 and 4 add no cycles: they sit between the MAC's accumulator register
and the output port, so a write-back beat is still one cycle per column. Only
stage 1 (one beat per cycle) and stage 2 (K cycles per group of `NUM_MACS`
columns) contribute to the cycle counts above.

The two layers of the network overlap at their boundary. Layer 2's LOAD
consumes activations in the same cycles layer 1's WRITEBACK produces them, so
the pair costs less than the two run separately. Measured at `NUM_MACS = 1`:

| Run                       | Busy cycles per inference |
|---------------------------|---------------------------|
| Layer 1 alone (16 → 32)   | 561                       |
| Layer 2 alone (32 → 10)   | 363                       |
| Sum, if not overlapped    | 924                       |
| Network (16 → 32 → 10)    | **892**                   |

The 32 cycles saved are exactly HIDDEN_SIZE: every activation is handed over
in the cycle it is produced, with one extra cycle at the start while layer 2
leaves IDLE.

### Latency and initiation interval

Latency is the busy window of one inference: 892 cycles for 16 → 32 → 10 at
`NUM_MACS = 1`, of which the last is DONE. The initiation interval depends on
how the host drives the input. If it waits for `done` before offering the next
X, the interval is the latency plus one, because the FSM passes through IDLE.
If it keeps the next X offered, layer 1 — a separate FSM — starts early and
the interval is shorter. Both are measured:

| NUM_MACS | Latency (busy) | Interval, waiting for `done` | Interval, X always offered | Slower layer alone | Throughput (inferences / 1000 cycles) |
|----------|----------------|------------------------------|----------------------------|--------------------|----------------------------------------|
| 1        | 892            | 893                          | **860**                    | 562                | 1.16                                   |
| 4        | 284            | 285                          | **252**                    | 178                | 3.97                                   |
| 8        | 188            | 189                          | **156**                    | 114                | 6.41                                   |
| 16       | 124            | 125                          | **92**                     | 82                 | 10.87                                  |
| 32       | 108            | 109                          | **76**                     | 76                 | 13.16                                  |

Throughput is one inference per interval, using the streamed figure. `make
perf` regenerates this sweep and writes [PERFORMANCE.md](PERFORMANCE.md) from
it. The same overlap on the small 4-3-2 network: 30 → 22 at one MAC and
19 → 14 at four, against slower-layer periods of 21 and 13.

Layer 1 runs ahead only until its own first write-back beat. There is no
buffer between the layers — layer 1 writes activations straight into layer 2's
operand memory — so it then stalls until layer 2 returns to LOAD. For
16-32-10 the gain is a steady 33 cycles at every width, which is the IDLE
cycle plus the X load plus the first group's accumulation. That is 3.7% of 893
at one MAC but 30% of 109 at thirty-two, and at thirty-two it is enough to
reach the slower layer's own period of 76 — fully pipelined, with layer 1
never idle. At sixteen MACs it comes close, 92 against a floor of 82.

The testbench measures this rather than assuming it: it offers the same X four
times without waiting for `done`, checks every logit of every inference, and
requires the interval to be steady, strictly shorter than waiting for `done`,
and never shorter than the slower layer running alone.

For the single matrix multiplier there is no such overlap — it is one FSM — and
`matrix_mult_tb` checks it: a source that offers the next inference's first
beat while the current one computes must see `s_ready` stay low, and the
spacing must be exactly the documented interval.

Weight loads are not in this path. One load of 834 cycles serves any number of
inferences that follow, so a run of *n* inferences costs 834 + 860·*n* cycles
at one MAC with the input kept fed, and 834 + 76·*n* at thirty-two. Before
Phase 8 the same run cost 1,725·*n*.

### MAC utilization

Each layer instantiates its own `NUM_MACS` MACs, so the network holds
2·`NUM_MACS` of them, and only one layer's array is active at a time. One
16 → 32 → 10 inference needs 16·32 + 32·10 = 832 multiply-accumulates.

Measured in steady state, over the interval with the input kept fed:

| NUM_MACS | MAC slots in the active array | Useful | Utilization |
|----------|-------------------------------|--------|-------------|
| 1        | 1 × 860 = 860                 | 832    | 96.7%       |
| 32       | 32 × 76 = 2,432               | 832    | 34.2%       |

At one MAC the array is busy almost all the time. At thirty-two it is idle for
two cycles in three: the input load, the activation handoff and the output
beats do not shrink with `NUM_MACS`, and layer 2 only has 10 output columns,
so 22 of its 32 MACs have nothing to do. This is the cost of the schedule, and
it is visible in the table above as the drop from 7.14× to 8.19× between 16
and 32 MACs.

### Hazards

- **Structural: no buffer between the layers.** Layer 1 writes each activation
  straight into layer 2's operand memory, so it can only run ahead of layer 2
  until its own first write-back beat, and then stalls. One activation buffer
  between them would let layer 1 finish an inference while layer 2 works on
  the previous one, bringing the interval down towards the slower layer's own
  period at every width, not just at thirty-two MACs. Not implemented.
- **Structural: one activation buffer per layer.** Each layer's operand memory
  holds one inference's inputs, so a layer cannot load the next while still
  reading the current. This is what bounds the run-ahead above.
- **Structural: one output beat per cycle.** The result port is one column
  wide whatever `NUM_MACS` is, so M·N write-back cycles never shrink. This is
  the floor the utilization table runs into.
- **Data: accumulator read-modify-write.** The MAC reads and writes its
  accumulator every cycle. `clear` asserted together with `en` loads the first
  product directly instead of adding it to a stale value, so consecutive
  groups need no idle cycle between them.
- **Data: the activation handoff.** Requantization is combinational, so layer
  1's output reaches layer 2's input in the same cycle. Layer 2 must already
  be in LOAD to accept it, and it is one cycle behind leaving IDLE — the `+1`
  in the cycle table. After that the handoff runs at one activation per cycle.
- **Control: no combinational handshake loops.** `s_ready` and `m_valid` are
  decoded from the registered state alone (Moore), so `s_ready` never depends
  on `s_valid` and `m_valid` never on `m_ready`. A stalled cycle therefore
  costs exactly one cycle, which the testbenches check cycle by cycle under
  random stalls.
- **Weight coherence is the host's job.** Weights are resident state. The FSM
  runs one job at a time, so a load cannot interleave with an inference, but
  nothing in the hardware records whether weights were ever loaded: reset
  returns the FSM to IDLE and leaves the operand memory untouched. A host that
  infers before loading gets arithmetic on whatever was there.

## Verification flow

Two complementary flows, both run by `make test`.

**Testbench-driven** (`tb/*_tb.sv`). Each module has a self-checking
testbench that generates its own stimulus, models the expected behaviour
cycle by cycle, and checks timing as well as values: exact busy-cycle counts
under random stalls, protocol rules on every clock edge, resets from every
state, and coverage bins that fail the run if a hole is left. These are what
catch a schedule or handshake bug. Several of them also write a results file
that `python/verify_results.py` checks against NumPy.

**Vector-driven** (`tb/*_vec_tb.sv`, `python/run_regression.py`). Python
generates the operands, computes the expected outputs with NumPy, writes a
vector file, runs the simulation, and compares. These testbenches invent
nothing and check no timing; their job is volume and Python-chosen
distributions. `make regression` runs 1,000 cases against each of two DUTs.

```
  python/run_regression.py
      |  numpy: random INT8 operands + directed edge cases
      v
  vectors/regression_<dut>_<sim>.txt        (plain integers, read with $fscanf)
      |
      v
  tb/nn_accelerator_vec_tb.sv  or  tb/matrix_mult_vec_tb.sv
      |  drives the weight-stationary interface: one weight load per group,
      |  then one inference per case
      v
  sim/regression_<dut>_<sim>.results.txt    (the format verify_results.py reads)
      |
      v
  python/run_regression.py  ->  numpy golden model  ->  exit 0 or nonzero
```

The two levels see different things. The network testbench exercises the
whole datapath but its hidden activations are ReLU'd and requantized to INT8,
so the accumulator's full width never reaches an output. The matrix-multiply
testbench sits one level down, where every element of C leaves as a raw
signed INT32 — the same 1,000 cases check 64,640 values there against 10,150
logits at the top.

Neither reaches the requantizer's sign behaviour: in the network ReLU keeps
its input non-negative, so an arithmetic and a logical right shift agree, and
the matrix multiplier has no requantizer. `tb/requant_tb.sv` drives it
directly with negative values, which is why the unit testbenches stay.
