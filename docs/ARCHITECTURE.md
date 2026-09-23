# Architecture

This document describes the hardware as built so far. It will grow as later
phases add the neural-network layer, parallel MACs and pipelining.

## Blocks

| Block          | File                  | Role                                                        |
|----------------|-----------------------|-------------------------------------------------------------|
| MAC            | `rtl/mac.sv`          | Signed INT8 × INT8 → INT32 multiply-accumulate, 1 per clock |
| ReLU           | `rtl/relu.sv`         | `y = x < 0 ? 0 : x` on INT32, combinational                 |
| Matrix multiply| `rtl/matrix_mult.sv`  | Streaming C = A × B datapath: operand memory + one MAC      |
| Controller     | `rtl/matmul_ctrl.sv`  | FSM that sequences a matrix-multiply job                    |
| Shared types   | `rtl/matmul_pkg.sv`   | Controller state encoding                                   |

## Matrix multiplier

`C = A × B`, where A (M×K) and B (K×N) are signed INT8 and C (M×N) is
signed INT32. The default is M = K = N = 8, and all three are parameters.

```
                      +--------------------------------+
 s_valid, s_ready <-->|          matmul_ctrl           |<--> m_valid, m_ready, m_last
                      | IDLE LOAD COMPUTE WRITEBACK .. |---> busy, done
                      +--------------------------------+
                        | ld_en      | i, j, k    | mac_en, mac_clear
                        v ld_idx     v            v
 s_data --------->[ operand memory ]-- A[i][k] -->[     ]
 (INT8)           [ A, then B:     ]              [ mac ]--- acc ---> m_data (INT32)
                  [ M*K + K*N B    ]-- B[k][j] -->[     ]
```

- **Operand memory.** A and B are stored back to back in one memory,
  M·K + K·N bytes (128 for 8×8×8). It has one write port, used while
  loading, and two read ports: A[i][k] at address `i*K + k`, and B[k][j] at
  address `M*K + k*N + j`.
- **One MAC.** Each element of C is a K-term dot product, computed in K
  consecutive cycles. The first term uses the MAC's clear+en load, so a new
  sum starts without an idle cycle.
- **No C buffer.** The MAC's accumulator register is the output data
  register. While an element waits to be accepted, the MAC is idle, and its
  accumulator holds the value steady. (Phase 3's first version kept a
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
          s_valid             last beat accepted          k == K-1
+------+ -------> +------+ ---------------------> +---------+ ------> +-----------+
| IDLE |          | LOAD |                        | COMPUTE |         | WRITEBACK |
+------+          +------+                        +---------+ <------ +-----------+
    ^                                                        m_ready,      |
    |                                                        more left     | m_ready,
    |               +------+                                               | last element
    +---------------| DONE |<----------------------------------------------+
                    +------+
```

| State     | Leaves when                                   | Goes to   | Counters |
|-----------|-----------------------------------------------|-----------|----------|
| IDLE      | `s_valid`                                     | LOAD      | all cleared |
| LOAD      | a beat is accepted and it was the last of M·K + K·N | COMPUTE | `ld_idx` + 1 per accepted beat |
| COMPUTE   | `k == K-1` (after K cycles)                   | WRITEBACK | `k` + 1, wrapping to 0 |
| WRITEBACK | `m_ready`, and more elements remain           | COMPUTE   | next (i, j), row-major |
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
| `m_last`    | WRITEBACK and (i, j) = (M-1, N-1)      |
| `busy`      | not IDLE                               |
| `done`      | DONE                                   |

IDLE deliberately does not accept data, so LOAD is the only state that
writes operands. This costs one cycle per job.

### Timing

The start of a job, with no stalls (8×8×8):

```
cycle      0      1      2    ...   128    129   ...   136    137    138   ...
state     IDLE   LOAD   LOAD  ...   LOAD  COMPUTE ... COMPUTE  WB   COMPUTE ...
s_valid    1      1      1           1      0
s_ready    0      1      1           1      0
beat              A00    A01         B77
k                                          0     ...    7            0
mac_clear                                  1                         1
m_valid                                                         1
m_data                                                        C00
```

The end of the job: C[7][7] is computed in cycles 696–703. Its write-back
beat is cycle 704, and DONE is cycle 705. In cycle 706 the next job's IDLE
cycle can start.

| Phase                       | Cycles (general)              | 8×8×8 |
|-----------------------------|-------------------------------|-------|
| IDLE, seeing `s_valid`      | 1                             | 1     |
| LOAD                        | M·K + K·N                     | 128   |
| COMPUTE + WRITEBACK         | M·N·(K + 1)                   | 576   |
| DONE                        | 1                             | 1     |
| **Job, back to back**       | **M·K + K·N + M·N·(K+1) + 2** | **706** |

- **Busy time.** `busy` is high for all but the IDLE cycle: 705 cycles for
  8×8×8.
- **Stalls.** Every cycle with `s_valid` low during LOAD, or `m_ready` low
  during WRITEBACK, adds exactly one cycle. Both testbenches check this
  cycle for cycle, under random stalls.
- **Measured job periods.** Back to back with no stalls: 706 cycles for
  8×8×8, 385 for 3×16×5, and 282 for 1×16×8. Each matches the formula.
- **Where the time goes (8×8×8).** Of the 706 cycles, the MAC is busy for
  M·N·K = 512 (72.5%). The rest is loading (128 cycles), write-back beats
  (64), and IDLE plus DONE (2). Overlapping these with computation, and
  adding MACs, is the subject of Phases 7 and 8.
