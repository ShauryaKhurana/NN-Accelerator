// =============================================================================
// nn_layer.sv - fully connected INT8 layer: Y = ReLU(X * W + B)
// =============================================================================
// X is a row vector of INPUT_SIZE signed INT8 activations, W is
// INPUT_SIZE x OUTPUT_SIZE signed INT8 weights, B is one signed INT32 bias per
// output, and Y is OUTPUT_SIZE signed INT32 values.
//
//   s_data -->[ matrix_mult (M=1, K=INPUT_SIZE, N=OUTPUT_SIZE) ]--+
//   (X then W, INT8)        X . W[:,j], INT32                     |
//                                                                 v
//                          B[j] ------------------------------>[ + ]
//                       (INT32, held steady)                      |
//                                                                 v
//                                                       [ relu (optional) ] --> m_data
//
// The multiplier from Phase 4 does the dot products: a 1 x K by K x N multiply
// is exactly this layer's X * W. Its output beats arrive in order, one per
// output channel, so a counter following the output handshakes picks the
// matching bias. Adding the bias and applying ReLU is combinational, so the
// layer adds no cycles to the multiplier's schedule.
//
// Streams (same protocol as matrix_mult; a beat transfers on a rising edge
// where valid and ready are both high):
//   input   INPUT_SIZE elements of X, then INPUT_SIZE*OUTPUT_SIZE elements of
//           W row-major (W[k][j] at k*OUTPUT_SIZE + j), one INT8 per beat
//   output  OUTPUT_SIZE elements of Y, one INT32 per beat, m_last on the last
//
// The bias is a parallel input rather than part of the stream: there is one
// per output channel, it is INT32 while the stream is INT8, and in a larger
// design it would sit in a small register file written once per layer. It
// must hold steady while the layer is busy. Element j is at
// bias_flat[j*ACC_WIDTH +: ACC_WIDTH].
//
// APPLY_RELU = 0 leaves the sum as it is, for an output layer that produces
// raw logits rather than activations.
//
// Numeric range: the dot product of K INT8 pairs fits comfortably in INT32
// (at most K * 2^14 in magnitude). Adding the bias wraps modulo 2^32 like any
// other two's-complement add, so a bias of magnitude up to
// 2^31 - 1 - K * 2^14 can never make the sum wrap; for K = 16 that is any
// bias within +-2,147,221,503. The testbench checks both sides of that edge.
// ReLU is applied after the bias, so Y is never negative.
//
// NUM_MACS output channels are computed in parallel (see matmul_ctrl.sv).
// Cycles per inference, back to back and without stalls:
//   INPUT_SIZE*(1 + OUTPUT_SIZE)                       load X and W
//   + ceil(OUTPUT_SIZE/NUM_MACS)*INPUT_SIZE            compute
//   + OUTPUT_SIZE                                      output beats
//   + 2                                                IDLE and DONE
//   = 282 for 16 inputs, 8 outputs and one MAC.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module nn_layer #(
    parameter int INPUT_SIZE  = 16,   // length of X, rows of W
    parameter int OUTPUT_SIZE = 8,    // columns of W, length of Y
    parameter int DATA_WIDTH  = 8,    // signed activation / weight width
    parameter int ACC_WIDTH   = 32,   // signed accumulator / bias / output width
    parameter bit APPLY_RELU  = 1'b1, // 0 for an output layer (raw logits)
    parameter int NUM_MACS    = 1     // output channels computed in parallel
) (
    input  logic                             clk,
    input  logic                             rst,      // synchronous, active-high
    // Input stream: X, then W
    input  logic                             s_valid,
    output logic                             s_ready,
    input  logic signed [DATA_WIDTH-1:0]     s_data,
    // Bias: one INT32 per output channel, steady while busy
    input  logic [OUTPUT_SIZE*ACC_WIDTH-1:0] bias_flat,
    // Output stream: Y
    output logic                             m_valid,
    input  logic                             m_ready,
    output logic signed [ACC_WIDTH-1:0]      m_data,
    output logic                             m_last,
    // Status
    output logic                             busy,
    output logic                             done      // one-cycle pulse after the last Y beat
);

    localparam int JW = (OUTPUT_SIZE > 1) ? $clog2(OUTPUT_SIZE) : 1;

    // -------------------------------------------------------------------------
    // Dot products: X * W
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] dot;        // X . W[:,j] for the current channel
    logic                        dot_valid;
    logic                        dot_last;

    matrix_mult #(
        .M          (1),
        .K          (INPUT_SIZE),
        .N          (OUTPUT_SIZE),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .NUM_MACS   (NUM_MACS)
    ) u_matmul (
        .clk     (clk),
        .rst     (rst),
        .s_valid (s_valid),
        .s_ready (s_ready),
        .s_data  (s_data),
        .m_valid (dot_valid),
        .m_ready (m_ready),
        .m_data  (dot),
        .m_last  (dot_last),
        .busy    (busy),
        .done    (done)
    );

    // -------------------------------------------------------------------------
    // Bias select: which output channel is being offered
    // -------------------------------------------------------------------------
    logic [JW-1:0] out_idx;

    always_ff @(posedge clk) begin
        if (rst) begin
            out_idx <= '0;
        end else if (dot_valid && m_ready) begin      // this channel was accepted
            if (dot_last) out_idx <= '0;              // next job starts at channel 0
            else          out_idx <= out_idx + JW'(1);
        end
    end

    logic signed [ACC_WIDTH-1:0] bias [OUTPUT_SIZE];

    for (genvar ch = 0; ch < OUTPUT_SIZE; ch++) begin : g_bias
        assign bias[ch] = bias_flat[ch*ACC_WIDTH +: ACC_WIDTH];
    end

    // -------------------------------------------------------------------------
    // Bias add and ReLU (combinational: no extra cycle)
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] biased;

    assign biased  = dot + bias[out_idx];
    assign m_valid = dot_valid;
    assign m_last  = dot_last;

    if (APPLY_RELU) begin : g_relu
        relu #(
            .WIDTH (ACC_WIDTH)
        ) u_relu (
            .x (biased),
            .y (m_data)
        );
    end else begin : g_no_relu
        assign m_data = biased;
    end

endmodule

`default_nettype wire
