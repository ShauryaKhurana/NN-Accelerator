// =============================================================================
// matrix_mult.sv - C = A x B on a single MAC (INT8 x INT8 -> INT32)
// =============================================================================
// The simplest correct architecture: A and B are captured into local
// registers, then one MAC (rtl/mac.sv) computes the M*N dot products one
// after another, one multiply-accumulate per clock.
//
//            start
//              |
//   a_flat -->[ A regs, M x K ]--- A[i][k] ---+
//                                             +-->[ mac ]-- acc -->[ C regs, M x N ]--> c_flat
//   b_flat -->[ B regs, K x N ]--- B[k][j] ---+       ^                 ^
//                                                     | en, clear       | write C[i][j]
//             [ sequencer: k fastest, then j, then i ]+-----------------+
//
// Schedule for M = K = N = 8:
//   * start is sampled on a rising edge ("edge 0") and A, B are captured there.
//   * Edges 1..512 each perform one multiply-accumulate. The first term of
//     every dot product is issued with clear+en, so the MAC starts a new sum
//     with no idle cycle between dot products.
//   * Each finished sum is written into C one cycle after its last term,
//     overlapping the next dot product's first term.
//   * The final write happens on edge 513, and done is high for the cycle
//     that follows.
//
//     latency         M*N*K + 1 cycles from the start edge to done     (513)
//     start-to-start  M*N*K + 2 cycles when runs are back to back      (514)
//     MAC active      M*N*K cycles                                     (512)
//
// Interface:
//   start   accepted only when idle (busy = 0); ignored while busy
//   busy    high from the cycle after start through the final C write
//   done    one-cycle pulse. c_flat then holds C until the next run's first
//           result overwrites it.
//   a_flat / b_flat / c_flat hold matrices row-major: element (r, c) of an
//   R x C matrix sits at bits [(r*C + c)*W +: W].
//
// Only control state is reset. The operand and result registers are data:
// they are always written before they are read, so resetting them would
// only cost area.
// The sequencer is a minimal set of counters; Phase 4 replaces it with a
// proper FSM and a valid/ready load interface.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module matrix_mult #(
    parameter int M          = 8,    // rows of A and C
    parameter int K          = 8,    // columns of A = rows of B (dot-product length)
    parameter int N          = 8,    // columns of B and C
    parameter int DATA_WIDTH = 8,    // signed operand width
    parameter int ACC_WIDTH  = 32    // signed accumulator / result width
) (
    input  logic                      clk,
    input  logic                      rst,      // synchronous, active-high
    input  logic                      start,    // begin C = A x B (ignored while busy)
    input  logic [M*K*DATA_WIDTH-1:0] a_flat,   // A, row-major, captured on start
    input  logic [K*N*DATA_WIDTH-1:0] b_flat,   // B, row-major, captured on start
    output logic                      busy,
    output logic                      done,     // one-cycle pulse: C is complete
    output logic [M*N*ACC_WIDTH-1:0]  c_flat    // C, row-major
);

    // Counter widths: at least 1 bit, so a dimension of 1 still has a counter
    localparam int IW = (M > 1) ? $clog2(M) : 1;
    localparam int KW = (K > 1) ? $clog2(K) : 1;
    localparam int JW = (N > 1) ? $clog2(N) : 1;
    localparam logic [IW-1:0] I_LAST = IW'(M - 1);
    localparam logic [KW-1:0] K_LAST = KW'(K - 1);
    localparam logic [JW-1:0] J_LAST = JW'(N - 1);

    // -------------------------------------------------------------------------
    // Storage
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] a_reg [M][K];
    logic signed [DATA_WIDTH-1:0] b_reg [K][N];
    logic signed [ACC_WIDTH-1:0]  c_reg [M][N];

    // -------------------------------------------------------------------------
    // Sequencer
    // -------------------------------------------------------------------------
    logic          computing;   // a multiply-accumulate is issued this cycle
    logic [IW-1:0] i;           // row of the dot product being computed
    logic [JW-1:0] j;           // column of the dot product being computed
    logic [KW-1:0] k;           // term within the dot product
    logic          wb_valid;    // the MAC holds a finished dot product ...
    logic          wb_last;     // ... and it is the final one, C[M-1][N-1]
    logic [IW-1:0] wb_i;        // where that dot product belongs in C
    logic [JW-1:0] wb_j;

    logic accept;               // start is taken this cycle
    logic last_term;            // final term of the final dot product
    logic mac_en;
    logic mac_clear;

    assign busy      = computing || wb_valid;
    assign accept    = start && !busy;
    assign last_term = computing && (k == K_LAST) && (j == J_LAST) && (i == I_LAST);
    assign mac_en    = computing;
    assign mac_clear = computing && (k == '0);   // first term: start a new sum

    always_ff @(posedge clk) begin
        if (rst) begin
            computing <= 1'b0;
            wb_valid  <= 1'b0;
            wb_last   <= 1'b0;
            done      <= 1'b0;
            i         <= '0;
            j         <= '0;
            k         <= '0;
        end else begin
            // A dot product is finished in the MAC one cycle after its last term
            wb_valid <= computing && (k == K_LAST);
            wb_last  <= last_term;
            done     <= wb_valid && wb_last;

            if (accept) begin
                computing <= 1'b1;
                i         <= '0;
                j         <= '0;
                k         <= '0;
            end else if (computing) begin
                // k runs fastest, then j, then i: C is produced in row-major order
                if (k != K_LAST) begin
                    k <= k + KW'(1);
                end else begin
                    k <= '0;
                    if (j != J_LAST) begin
                        j <= j + JW'(1);
                    end else begin
                        j <= '0;
                        if (i != I_LAST) i <= i + IW'(1);
                        else             computing <= 1'b0;   // last term issued
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Datapath: operand capture, operand select, MAC, result write-back
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] mac_a;
    logic signed [DATA_WIDTH-1:0] mac_b;
    logic signed [ACC_WIDTH-1:0]  mac_acc;

    always_ff @(posedge clk) begin
        if (accept) begin
            for (int r = 0; r < M; r++)
                for (int c = 0; c < K; c++)
                    a_reg[r][c] <= a_flat[(r*K + c)*DATA_WIDTH +: DATA_WIDTH];
            for (int r = 0; r < K; r++)
                for (int c = 0; c < N; c++)
                    b_reg[r][c] <= b_flat[(r*N + c)*DATA_WIDTH +: DATA_WIDTH];
        end

        wb_i <= i;
        wb_j <= j;
        if (wb_valid) c_reg[wb_i][wb_j] <= mac_acc;
    end

    assign mac_a = a_reg[i][k];
    assign mac_b = b_reg[k][j];

    mac #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_mac (
        .clk   (clk),
        .rst   (rst),
        .en    (mac_en),
        .clear (mac_clear),
        .a     (mac_a),
        .b     (mac_b),
        .acc   (mac_acc)
    );

    for (genvar row = 0; row < M; row++) begin : g_c_row
        for (genvar col = 0; col < N; col++) begin : g_c_col
            assign c_flat[(row*N + col)*ACC_WIDTH +: ACC_WIDTH] = c_reg[row][col];
        end
    end

endmodule

`default_nettype wire
