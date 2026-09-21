// =============================================================================
// matrix_mult.sv - streaming C = A x B on a single MAC (INT8 x INT8 -> INT32)
// =============================================================================
// Datapath of the matrix multiplier. The control FSM is rtl/matmul_ctrl.sv.
//
//                        +--------------------------------+
//   s_valid, s_ready <-->|          matmul_ctrl           |<--> m_valid, m_ready, m_last
//                        | IDLE LOAD COMPUTE WRITEBACK .. |---> busy, done
//                        +--------------------------------+
//                          | ld_en      | i, j, k    | mac_en, mac_clear
//                          v ld_idx     v            v
//   s_data ---------->[ operand memory ]-- A[i][k] -->[     ]
//   (INT8)            [ A, then B:     ]              [ mac ]--- acc ---> m_data (INT32)
//                     [ M*K + K*N B    ]-- B[k][j] -->[     ]
//
// Stream protocol (both directions): a beat transfers on a rising edge where
// valid and ready are both high.
//   input   M*K elements of A, then K*N elements of B, each row-major, one
//           signed INT8 per beat on s_data
//   output  M*N elements of C, row-major, one signed INT32 per beat on m_data.
//           m_last marks the final element. While m_valid is high and m_ready
//           is low, m_data and m_last hold steady.
//
// C streams straight from the MAC's accumulator. In WRITEBACK the MAC is
// idle and holds the finished sum, so no C buffer is needed.
// Cycles per job (8x8x8, no stalls): 706; see matmul_ctrl.sv for the breakdown.
// The operand memory is data, not control, so it is not reset.
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

    logic          ld_en;
    logic [LW-1:0] ld_idx;
    logic          mac_en;
    logic          mac_clear;
    logic [IW-1:0] i;
    logic [JW-1:0] j;
    logic [KW-1:0] k;

    matmul_ctrl #(
        .M (M),
        .K (K),
        .N (N)
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
        .j         (j),
        .k         (k)
    );

    // -------------------------------------------------------------------------
    // Operand memory: A (row-major) at [0, M*K), then B (row-major)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] op_mem [LOAD_BEATS];
    logic        [LW-1:0]         a_addr;
    logic        [LW-1:0]         b_addr;
    logic signed [DATA_WIDTH-1:0] mac_a;
    logic signed [DATA_WIDTH-1:0] mac_b;

    always_ff @(posedge clk) begin
        if (ld_en) op_mem[ld_idx] <= s_data;
    end

    assign a_addr = LW'(i) * LW'(K) + LW'(k);                  // A[i][k]
    assign b_addr = LW'(M*K) + LW'(k) * LW'(N) + LW'(j);       // B[k][j]
    assign mac_a  = op_mem[a_addr];
    assign mac_b  = op_mem[b_addr];

    // -------------------------------------------------------------------------
    // MAC: its accumulator is the output data register
    // -------------------------------------------------------------------------
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
        .acc   (m_data)
    );

endmodule

`default_nettype wire
