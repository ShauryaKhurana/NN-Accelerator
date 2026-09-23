// =============================================================================
// nn_accelerator.sv - two-layer INT8 network
// =============================================================================
//   INPUT_SIZE inputs -> [ layer 1 + ReLU ] -> requantize to INT8
//                     -> [ layer 2 ] -> OUTPUT_SIZE logits (INT32)
//
// The default shape is 16 -> 32 -> 10.
//
//   s_data ---+--->[ nn_layer 1: INPUT_SIZE -> HIDDEN_SIZE, ReLU ]
//   (INT8)    |            |
//   X, W1, W2 |            | activations (INT32, non-negative)
//             |            v
//             |      [ requant: >>> SHIFT, saturate ]
//             |            | INT8
//             |            v
//             +--->[ nn_layer 2: HIDDEN_SIZE -> OUTPUT_SIZE, no ReLU ]---> m_data
//               W2                                                         (INT32 logits)
//
// Why requantize: layer 1 accumulates in INT32, but layer 2's multiplier takes
// INT8. An arithmetic right shift with saturation brings the activations back
// into range; see rtl/requant.sv. Everything stays integer, so the hardware
// and the NumPy golden model agree exactly.
//
// Stream protocol (same valid/ready rules as the layers):
//   input   X (INPUT_SIZE), then W1 (INPUT_SIZE*HIDDEN_SIZE), then
//           W2 (HIDDEN_SIZE*OUTPUT_SIZE), all row-major, one INT8 per beat
//   output  OUTPUT_SIZE logits, one INT32 per beat, m_last on the last
//   bias    bias1_flat and bias2_flat are parallel inputs, one INT32 per
//           output channel of their layer, steady while busy
//
// Routing: a beat counter decides where each input beat goes. The first
// INPUT_SIZE + INPUT_SIZE*HIDDEN_SIZE beats feed layer 1. Layer 2 takes its
// HIDDEN_SIZE activations from layer 1 (through the requantizer) and only
// then accepts the W2 beats from the host, so `s_ready` stays low for W2
// until the activations have gone through.
//
// Layer 2 loads its activations while layer 1 is still computing the later
// ones, which is why the total is less than the sum of two separate layers.
// Cycles per inference, no stalls (16 -> 32 -> 10 in brackets):
//   1                              IDLE, seeing the first beat
//   + INPUT_SIZE*(1 + HIDDEN_SIZE) layer 1 loads X and W1          [528]
//   + HIDDEN_SIZE*(INPUT_SIZE+1)+1 layer 1 computes; the first
//                                  activation waits one cycle
//                                  for layer 2 to leave IDLE       [545]
//   + HIDDEN_SIZE*OUTPUT_SIZE      layer 2 loads W2                [320]
//   + OUTPUT_SIZE*(HIDDEN_SIZE+1)  layer 2 computes                [330]
//   + 1                            DONE                            [1]
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module nn_accelerator #(
    parameter int INPUT_SIZE  = 16,
    parameter int HIDDEN_SIZE = 32,
    parameter int OUTPUT_SIZE = 10,
    parameter int DATA_WIDTH  = 8,    // signed activation / weight width
    parameter int ACC_WIDTH   = 32,   // signed accumulator / bias / logit width
    parameter int SHIFT       = 8     // requantization shift between the layers
) (
    input  logic                             clk,
    input  logic                             rst,      // synchronous, active-high
    // Input stream: X, then W1, then W2
    input  logic                             s_valid,
    output logic                             s_ready,
    input  logic signed [DATA_WIDTH-1:0]     s_data,
    // Biases, steady while busy
    input  logic [HIDDEN_SIZE*ACC_WIDTH-1:0] bias1_flat,
    input  logic [OUTPUT_SIZE*ACC_WIDTH-1:0] bias2_flat,
    // Output stream: logits
    output logic                             m_valid,
    input  logic                             m_ready,
    output logic signed [ACC_WIDTH-1:0]      m_data,
    output logic                             m_last,
    // Status
    output logic                             busy,
    output logic                             done      // one-cycle pulse after the last logit
);

    localparam int L1_BEATS   = INPUT_SIZE + INPUT_SIZE*HIDDEN_SIZE;   // X and W1
    localparam int W2_BEATS   = HIDDEN_SIZE*OUTPUT_SIZE;
    localparam int HOST_BEATS = L1_BEATS + W2_BEATS;
    localparam int HC_W       = $clog2(HOST_BEATS + 1);
    localparam int AC_W       = $clog2(HIDDEN_SIZE + 1);

    // -------------------------------------------------------------------------
    // Beat routing
    // -------------------------------------------------------------------------
    logic [HC_W-1:0] host_cnt;   // input beats taken from the host this inference
    logic [AC_W-1:0] act_cnt;    // activations handed from layer 1 to layer 2
    logic            host_to_l1; // host beats still belong to layer 1
    logic            acts_left;  // layer 2 is still taking activations

    assign host_to_l1 = (host_cnt < HC_W'(L1_BEATS));
    assign acts_left  = (act_cnt  < AC_W'(HIDDEN_SIZE));

    logic                        l1_s_valid, l1_s_ready;
    logic                        l1_m_valid, l1_m_ready, l1_m_last, l1_busy, l1_done;
    logic signed [ACC_WIDTH-1:0] l1_m_data;
    logic signed [DATA_WIDTH-1:0] act;           // requantized activation
    logic                        l2_s_valid, l2_s_ready, l2_busy;
    logic signed [DATA_WIDTH-1:0] l2_s_data;

    assign l1_s_valid = s_valid && host_to_l1;
    assign l1_m_ready = acts_left && l2_s_ready;

    assign l2_s_valid = acts_left ? l1_m_valid : (s_valid && !host_to_l1);
    assign l2_s_data  = acts_left ? act        : s_data;

    assign s_ready    = host_to_l1 ? l1_s_ready : (!acts_left && l2_s_ready);

    always_ff @(posedge clk) begin
        if (rst) begin
            host_cnt <= '0;
            act_cnt  <= '0;
        end else if (done) begin                       // start the next inference clean
            host_cnt <= '0;
            act_cnt  <= '0;
        end else begin
            if (s_valid && s_ready)       host_cnt <= host_cnt + HC_W'(1);
            if (l1_m_valid && l1_m_ready) act_cnt  <= act_cnt  + AC_W'(1);
        end
    end

    // -------------------------------------------------------------------------
    // Layer 1: INPUT_SIZE -> HIDDEN_SIZE, with ReLU
    // -------------------------------------------------------------------------
    nn_layer #(
        .INPUT_SIZE  (INPUT_SIZE),
        .OUTPUT_SIZE (HIDDEN_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .APPLY_RELU  (1'b1)
    ) u_layer1 (
        .clk       (clk),
        .rst       (rst),
        .s_valid   (l1_s_valid),
        .s_ready   (l1_s_ready),
        .s_data    (s_data),
        .bias_flat (bias1_flat),
        .m_valid   (l1_m_valid),
        .m_ready   (l1_m_ready),
        .m_data    (l1_m_data),
        .m_last    (l1_m_last),
        .busy      (l1_busy),
        .done      (l1_done)
    );

    // -------------------------------------------------------------------------
    // Requantize the activations to INT8
    // -------------------------------------------------------------------------
    requant #(
        .IN_WIDTH  (ACC_WIDTH),
        .OUT_WIDTH (DATA_WIDTH),
        .SHIFT     (SHIFT)
    ) u_requant (
        .x (l1_m_data),
        .y (act)
    );

    // -------------------------------------------------------------------------
    // Layer 2: HIDDEN_SIZE -> OUTPUT_SIZE, no ReLU (raw logits)
    // -------------------------------------------------------------------------
    nn_layer #(
        .INPUT_SIZE  (HIDDEN_SIZE),
        .OUTPUT_SIZE (OUTPUT_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .APPLY_RELU  (1'b0)
    ) u_layer2 (
        .clk       (clk),
        .rst       (rst),
        .s_valid   (l2_s_valid),
        .s_ready   (l2_s_ready),
        .s_data    (l2_s_data),
        .bias_flat (bias2_flat),
        .m_valid   (m_valid),
        .m_ready   (m_ready),
        .m_data    (m_data),
        .m_last    (m_last),
        .busy      (l2_busy),
        .done      (done)
    );

    assign busy = l1_busy || l2_busy;

    // l1_m_last and l1_done are not needed: the activation counter tracks
    // layer 1's output, and the inference ends with layer 2's done.
    logic unused_ok;
    assign unused_ok = l1_m_last & l1_done;

endmodule

`default_nettype wire
