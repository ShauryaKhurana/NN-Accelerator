// =============================================================================
// matmul_ctrl.sv - control FSM for the streaming matrix multiplier
// =============================================================================
// Runs one job at a time. A job is either
//   * a weight load: K*N beats into the B region of the operand memory, no
//     compute and no output. The weights stay there for later jobs.
//   * an inference: M*K beats of A, then each element of C computed on the
//     MACs and handed to the output stream.
// load_weights, sampled with the first beat, says which kind it is. Weights
// are loaded once and reused by every inference that follows, so an inference
// pays only for its own operands.
// The datapath (operand memory, MACs) lives in rtl/matrix_mult.sv.
//
//            s_valid        last A beat (inference)            k == K-1
//  +------+ -------> +------+ ---------------------> +---------+ ------> +-----------+
//  | IDLE |          | LOAD |                        | COMPUTE |         | WRITEBACK |
//  +------+          +------+                        +---------+ <------ +-----------+
//      ^                 |                                      m_ready,        |
//      |                 | last B beat (weight load)            more left       | m_ready,
//      |                 v                                                      | last element
//      |             +------+                                                   |
//      +-------------| DONE |<--------------------------------------------------+
//                    +------+
//
//  state      | leaves when                    | goes to   | counters
//  -----------+--------------------------------+-----------+------------------------------
//  IDLE       | s_valid                        | LOAD      | all cleared; the job kind
//             |                                |           | is latched from load_weights
//  LOAD       | last beat of a weight load     | DONE      | ld_idx + 1 per accepted beat
//             | (K*N beats)                    |           |
//  LOAD       | last beat of an inference's A  | COMPUTE   |
//             | (M*K beats)                    |           |
//  COMPUTE    | k == K-1 (after K cycles)      | WRITEBACK | k + 1, wrapping to 0
//  WRITEBACK  | m_ready, group not streamed    | WRITEBACK | wb_sel + 1
//  WRITEBACK  | m_ready, group done, more left | COMPUTE   | next group, then next row
//  WRITEBACK  | m_ready, last element          | DONE      |
//  DONE       | always (one cycle)             | IDLE      |
//  Each state stays put until its condition holds. Reset returns to IDLE
//  from any state.
//
// NUM_MACS MACs work on one group of output columns at a time: all of them
// see the same A[i][k], and MAC p handles column j_base + p. A group needs K
// cycles whatever its width, and is then streamed out one column per cycle,
// so the compute time falls with NUM_MACS while the output beats do not:
//
//   compute + write-back cycles = M * ceil(N/NUM_MACS) * K + M * N
//
// With NUM_MACS = 1 this is the earlier M*N*(K+1). When N is not a multiple
// of NUM_MACS the last group is short, and only its valid columns are
// streamed out.
//
// Outputs are decoded from the registered state and counters only (Moore),
// so s_ready and m_valid never depend combinationally on s_valid / m_ready:
//   s_ready   = LOAD                  ld_en     = LOAD and s_valid
//   mac_en    = COMPUTE               mac_clear = COMPUTE and k == 0
//   m_valid   = WRITEBACK             m_last    = WRITEBACK and the element is
//   busy      = not IDLE                          C[M-1][N-1]
//   done      = DONE (one-cycle pulse)
//   i, j_base, k select the operands; wb_sel picks the MAC being streamed out
//
// Cycles with no stalls, 8x8x8 in brackets:
//   weight load   1 + K*N + 1                                        [66]
//   inference     1 + M*K + M*ceil(N/NUM_MACS)*K + M*N + 1
//                                      [642 with one MAC, 194 with eight]
// Weights are paid for once; each inference after that costs only its own A
// beats, the compute, and its output beats. busy is high for every cycle of a
// job except the IDLE one, and each cycle with s_valid low in LOAD, or
// m_ready low in WRITEBACK, adds exactly one more.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module matmul_ctrl #(
    parameter int M  = 8,    // rows of A and C
    parameter int K  = 8,    // dot-product length
    parameter int N  = 8,    // columns of B and C
    parameter int NUM_MACS = 1,   // output columns computed in parallel
    // Counter widths, derived from the dimensions (do not override)
    parameter int LW = $clog2(M*K + K*N),
    parameter int IW = (M > 1) ? $clog2(M) : 1,
    parameter int JW = (N > 1) ? $clog2(N) : 1,
    parameter int KW = (K > 1) ? $clog2(K) : 1,
    parameter int PW = (NUM_MACS > 1) ? $clog2(NUM_MACS) : 1
) (
    input  logic          clk,
    input  logic          rst,          // synchronous, active-high
    // Job kind, sampled with the first beat: 1 = load weights, 0 = inference
    input  logic          load_weights,
    // Input stream handshake (the data goes straight to the datapath)
    input  logic          s_valid,
    output logic          s_ready,
    // Output stream handshake (the data comes from the datapath)
    output logic          m_valid,
    input  logic          m_ready,
    output logic          m_last,
    // Status
    output logic          busy,
    output logic          done,
    // Datapath control
    output logic          ld_en,       // store the current input beat ...
    output logic [LW-1:0] ld_idx,      // ... at this operand-memory address
    output logic          mac_en,
    output logic          mac_clear,
    output logic [IW-1:0] i,           // row of C being computed
    output logic [JW-1:0] j_base,      // first column of the group being computed
    output logic [KW-1:0] k,           // term within the dot products
    output logic [PW-1:0] wb_sel       // MAC whose result is being streamed out
);

    import matmul_pkg::*;

    localparam int              A_BEATS    = M*K;              // one inference's A
    localparam int              W_BEATS    = K*N;              // one weight load
    localparam int              GROUPS     = (N + NUM_MACS - 1) / NUM_MACS;
    localparam logic [LW-1:0]   A_LAST     = LW'(A_BEATS - 1);          // last A address
    localparam logic [LW-1:0]   W_LAST     = LW'(A_BEATS + W_BEATS - 1); // last weight address
    localparam logic [IW-1:0]   I_LAST     = IW'(M - 1);
    localparam logic [JW-1:0]   J_LAST_BASE = JW'((GROUPS - 1) * NUM_MACS);
    localparam logic [KW-1:0]   K_LAST     = KW'(K - 1);

    matmul_state_t state;
    matmul_state_t state_next;
    logic          weight_job;     // this job is a weight load
    logic          ld_last;        // this beat is the job's last
    logic [PW-1:0] wb_last;        // last valid MAC in this group (the tail group is short)
    logic          group_done;     // this write-back beat finishes the group
    logic          last_elem;      // the element being written back is C[M-1][N-1]

    assign wb_last    = (j_base == J_LAST_BASE) ? PW'(N - 1 - ((GROUPS - 1) * NUM_MACS))
                                                : PW'(NUM_MACS - 1);
    assign group_done = (wb_sel == wb_last);
    assign ld_last    = weight_job ? (ld_idx == W_LAST) : (ld_idx == A_LAST);
    assign last_elem  = (i == I_LAST) && (j_base == J_LAST_BASE) && group_done;

    // -------------------------------------------------------------------------
    // Next-state logic
    // -------------------------------------------------------------------------
    always_comb begin
        state_next = state;
        case (state)
            S_IDLE:      if (s_valid)                      state_next = S_LOAD;
            S_LOAD:      if (s_valid && ld_last) begin
                             if (weight_job)                state_next = S_DONE;
                             else                           state_next = S_COMPUTE;
                         end
            S_COMPUTE:   if (k == K_LAST)                  state_next = S_WRITEBACK;
            S_WRITEBACK: if (m_ready) begin
                             if (last_elem)                state_next = S_DONE;
                             else if (group_done)          state_next = S_COMPUTE;
                         end
            S_DONE:                                        state_next = S_IDLE;
            default:                                       state_next = S_IDLE;   // unused encodings
        endcase
    end

    // -------------------------------------------------------------------------
    // State register and counters
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state  <= S_IDLE;
            ld_idx     <= '0;
            i          <= '0;
            j_base     <= '0;
            k          <= '0;
            wb_sel     <= '0;
            weight_job <= 1'b0;
        end else begin
            state <= state_next;
            case (state)
                S_IDLE: begin
                    // A weight load writes the B region, an inference the A
                    // region; either way the first beat's address is set here.
                    weight_job <= load_weights;
                    ld_idx     <= load_weights ? LW'(A_BEATS) : '0;
                    i          <= '0;
                    j_base     <= '0;
                    k          <= '0;
                    wb_sel     <= '0;
                end
                S_LOAD:
                    if (s_valid) ld_idx <= ld_idx + LW'(1);
                S_COMPUTE:
                    k <= (k == K_LAST) ? '0 : k + KW'(1);
                S_WRITEBACK:
                    if (m_ready && !last_elem) begin
                        if (!group_done) begin             // next column of this group
                            wb_sel <= wb_sel + PW'(1);
                        end else begin                     // next group, then next row
                            wb_sel <= '0;
                            if (j_base == J_LAST_BASE) begin
                                j_base <= '0;
                                i      <= i + IW'(1);
                            end else begin
                                j_base <= j_base + JW'(NUM_MACS);
                            end
                        end
                    end
                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Outputs
    // -------------------------------------------------------------------------
    assign s_ready   = (state == S_LOAD);
    assign ld_en     = s_ready && s_valid;
    assign mac_en    = (state == S_COMPUTE);
    assign mac_clear = mac_en && (k == '0);
    assign m_valid   = (state == S_WRITEBACK);
    assign m_last    = m_valid && last_elem;
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule

`default_nettype wire
