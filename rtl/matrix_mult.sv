// =============================================================================
// matrix_mult.sv - streaming C = A x B on NUM_MACS MACs (INT8 x INT8 -> INT32)
// =============================================================================
// Datapath of the matrix multiplier. The control FSM is rtl/matmul_ctrl.sv.
//
//                        +--------------------------------+
//   s_valid, s_ready <-->|          matmul_ctrl           |<--> m_valid, m_ready, m_last
//                        | IDLE LOAD COMPUTE WRITEBACK .. |---> busy, done
//                        +--------------------------------+
//                          | ld_en   | i, j_base, k, wb_sel | mac_en, mac_clear
//                          v ld_idx   v                      v
//   s_data ---------->[ operand memory ]-- A[i][k] ------> broadcast to every MAC
//   (INT8)            [ A, then B:     ]-- B[k][j_base+p] -> MAC p
//                     [ M*K + K*N B    ]   NUM_MACS accumulators --mux--> m_data
//
// Stream protocol (both directions): a beat transfers on a rising edge where
// valid and ready are both high.
//   input   M*K elements of A, then K*N elements of B, each row-major, one
//           signed INT8 per beat on s_data
//   output  M*N elements of C, row-major, one signed INT32 per beat on m_data.
//           m_last marks the final element. While m_valid is high and m_ready
//           is low, m_data and m_last hold steady.
//
// NUM_MACS MACs compute one group of output columns at a time: they all see
// the same A[i][k], and MAC p reads B[k][j_base + p]. The group's results are
// streamed out one per cycle, picked by wb_sel. NUM_MACS = 1 is a single MAC
// and the original schedule.
//
// C streams straight from a MAC's accumulator. In WRITEBACK the MACs are idle
// and hold their finished sums, so no C buffer is needed.
// Cycles per job (8x8x8, no stalls): 706 with one MAC, 258 with eight; see
// matmul_ctrl.sv for the breakdown.
// The operand memory is data, not control, so it is not reset.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module matrix_mult #(
    parameter int M          = 8,    // rows of A and C
    parameter int K          = 8,    // columns of A = rows of B (dot-product length)
    parameter int N          = 8,    // columns of B and C
    parameter int DATA_WIDTH = 8,    // signed operand width
    parameter int ACC_WIDTH  = 32,   // signed accumulator / result width
    parameter int NUM_MACS   = 1     // output columns computed in parallel
) (
    input  logic                         clk,
    input  logic                         rst,       // synchronous, active-high
    // Input stream: A, then B
    input  logic                         s_valid,
    output logic                         s_ready,
    input  logic signed [DATA_WIDTH-1:0] s_data,
    // Output stream: C
    output logic                         m_valid,
    input  logic                         m_ready,
    output logic signed [ACC_WIDTH-1:0]  m_data,
    output logic                         m_last,
    // Status
    output logic                         busy,
    output logic                         done       // one-cycle pulse after the last C beat
);

    localparam int LOAD_BEATS = M*K + K*N;
    localparam int LW = $clog2(LOAD_BEATS);
    localparam int IW = (M > 1) ? $clog2(M) : 1;
    localparam int JW = (N > 1) ? $clog2(N) : 1;
    localparam int KW = (K > 1) ? $clog2(K) : 1;
    localparam int PW = (NUM_MACS > 1) ? $clog2(NUM_MACS) : 1;

    logic          ld_en;
    logic [LW-1:0] ld_idx;
    logic          mac_en;
    logic          mac_clear;
    logic [IW-1:0] i;
    logic [JW-1:0] j_base;
    logic [KW-1:0] k;
    logic [PW-1:0] wb_sel;

    matmul_ctrl #(
        .M        (M),
        .K        (K),
        .N        (N),
        .NUM_MACS (NUM_MACS)
    ) u_ctrl (
        .clk       (clk),
        .rst       (rst),
        .s_valid   (s_valid),
        .s_ready   (s_ready),
        .m_valid   (m_valid),
        .m_ready   (m_ready),
        .m_last    (m_last),
        .busy      (busy),
        .done      (done),
        .ld_en     (ld_en),
        .ld_idx    (ld_idx),
        .mac_en    (mac_en),
        .mac_clear (mac_clear),
        .i         (i),
        .j_base    (j_base),
        .k         (k),
        .wb_sel    (wb_sel)
    );

    // -------------------------------------------------------------------------
    // Operand memory: A (row-major) at [0, M*K), then B (row-major)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] op_mem [LOAD_BEATS];
    logic        [LW-1:0]         a_addr;
    logic signed [DATA_WIDTH-1:0] mac_a;
    logic signed [ACC_WIDTH-1:0]  mac_acc [NUM_MACS];

    always_ff @(posedge clk) begin
        if (ld_en) op_mem[ld_idx] <= s_data;
    end

    assign a_addr = LW'(i) * LW'(K) + LW'(k);                  // A[i][k], broadcast
    assign mac_a  = op_mem[a_addr];

    // -------------------------------------------------------------------------
    // One MAC per output column of the group. Their accumulators are the
    // output data registers; wb_sel picks the one being streamed out.
    // -------------------------------------------------------------------------
    for (genvar p = 0; p < NUM_MACS; p++) begin : g_mac
        logic [JW-1:0]                col;
        logic [LW-1:0]                b_addr;
        logic signed [DATA_WIDTH-1:0] mac_b;

        // In a short tail group the spare MACs repeat the last column, so the
        // address stays in range; their results are never streamed out.
        assign col    = (int'(j_base) + p < N) ? JW'(int'(j_base) + p) : JW'(N - 1);
        assign b_addr = LW'(M*K) + LW'(k) * LW'(N) + LW'(col);       // B[k][col]
        assign mac_b  = op_mem[b_addr];

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
            .acc   (mac_acc[p])
        );
    end

    assign m_data = mac_acc[wb_sel];

endmodule

`default_nettype wire
