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
// Weights stay resident. There are two kinds of job, selected by the
// load_weights input, which the host holds steady for the whole job:
//
//   load_weights = 1   W1 (INPUT_SIZE*HIDDEN_SIZE beats) followed by
//                      W2 (HIDDEN_SIZE*OUTPUT_SIZE beats), row-major. A beat
//                      counter routes the first group to layer 1 and the rest
//                      to layer 2. Nothing comes out.
//   load_weights = 0   one inference: X (INPUT_SIZE beats) into layer 1. Its
//                      activations flow through the requantizer into layer 2,
//                      whose logits leave on the output stream.
//
//   output  OUTPUT_SIZE logits, one INT32 per beat, m_last on the last
//   bias    bias1_flat and bias2_flat are parallel inputs, one INT32 per
//           output channel of their layer, steady while busy
//
// The two layers overlap. Within one inference, layer 2 takes each activation
// in the cycle layer 1 produces it, so the pair costs less than the two run
// separately. Across inferences, layer 1 can start the next X while layer 2
// is still computing, but only until its own first write-back: there is no
// buffer between the layers, so layer 1 then stalls until layer 2 is back in
// LOAD. The gain is therefore bounded, not the full "slowest stage" rate.
//
// Cycles with no stalls (16 -> 32 -> 10, one MAC / eight MACs):
//   weight load          INPUT*HIDDEN + 1 + HIDDEN*OUTPUT + 1        [834]
//   one inference, busy  INPUT + ceil(HIDDEN/P)*INPUT + HIDDEN + 1
//                        + ceil(OUTPUT/P)*HIDDEN + OUTPUT + 1  [892 / 188]
//   host waits for done  the above + 1                         [893 / 189]
//   next X always offered, measured                            [860 / 156]
// See docs/ARCHITECTURE.md for the stage breakdown and the hazards.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module nn_accelerator #(
    parameter int INPUT_SIZE  = 16,
    parameter int HIDDEN_SIZE = 32,
    parameter int OUTPUT_SIZE = 10,
    parameter int DATA_WIDTH  = 8,    // signed activation / weight width
    parameter int ACC_WIDTH   = 32,   // signed accumulator / bias / logit width
    parameter int SHIFT       = 8,    // requantization shift between the layers
    parameter int NUM_MACS    = 1     // output channels computed in parallel, per layer
) (
    input  logic                             clk,
    input  logic                             rst,      // synchronous, active-high
    // Job kind, held steady for the whole job: 1 = load W1 then W2, 0 = inference
    input  logic                             load_weights,
    // Input stream: weights, or one X vector
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

    localparam int W1_BEATS = INPUT_SIZE*HIDDEN_SIZE;
    localparam int W2_BEATS = HIDDEN_SIZE*OUTPUT_SIZE;
    localparam int W_BEATS  = W1_BEATS + W2_BEATS;
    localparam int WC_W     = $clog2(W_BEATS + 1);

    // -------------------------------------------------------------------------
    // Beat routing
    // -------------------------------------------------------------------------
    // During a weight load, a counter splits the stream: the first W1_BEATS go
    // to layer 1, the rest to layer 2. It resets whenever load_weights is low,
    // so an inference always starts the count clean.
    logic [WC_W-1:0] w_cnt;
    logic            w_to_l1;

    assign w_to_l1 = load_weights && (w_cnt < WC_W'(W1_BEATS));

    logic                         l1_s_valid, l1_s_ready;
    logic                         l1_m_valid, l1_m_ready, l1_m_last, l1_busy, l1_done;
    logic signed [ACC_WIDTH-1:0]  l1_m_data;
    logic signed [DATA_WIDTH-1:0] act;           // requantized activation
    logic                         l2_s_valid, l2_s_ready, l2_busy;
    logic signed [DATA_WIDTH-1:0] l2_s_data;

    assign l1_s_valid = s_valid && (load_weights ? w_to_l1 : 1'b1);
    assign l1_m_ready = !load_weights && l2_s_ready;

    assign l2_s_valid = load_weights ? (s_valid && !w_to_l1) : l1_m_valid;
    assign l2_s_data  = load_weights ? s_data : act;

    assign s_ready    = load_weights ? (w_to_l1 ? l1_s_ready : l2_s_ready) : l1_s_ready;

    always_ff @(posedge clk) begin
        if (rst || !load_weights) w_cnt <= '0;
        else if (s_valid && s_ready) w_cnt <= w_cnt + WC_W'(1);
    end

    // -------------------------------------------------------------------------
    // Layer 1: INPUT_SIZE -> HIDDEN_SIZE, with ReLU
    // -------------------------------------------------------------------------
    nn_layer #(
        .INPUT_SIZE  (INPUT_SIZE),
        .OUTPUT_SIZE (HIDDEN_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .APPLY_RELU  (1'b1),
        .NUM_MACS    (NUM_MACS)
    ) u_layer1 (
        .clk          (clk),
        .rst          (rst),
        .load_weights (load_weights),
        .s_valid      (l1_s_valid),
        .s_ready      (l1_s_ready),
        .s_data       (s_data),
        .bias_flat    (bias1_flat),
        .m_valid      (l1_m_valid),
        .m_ready      (l1_m_ready),
        .m_data       (l1_m_data),
        .m_last       (l1_m_last),
        .busy         (l1_busy),
        .done         (l1_done)
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
        .APPLY_RELU  (1'b0),
        .NUM_MACS    (NUM_MACS)
    ) u_layer2 (
        .clk          (clk),
        .rst          (rst),
        .load_weights (load_weights),
        .s_valid      (l2_s_valid),
        .s_ready      (l2_s_ready),
        .s_data       (l2_s_data),
        .bias_flat    (bias2_flat),
        .m_valid      (m_valid),
        .m_ready      (m_ready),
        .m_data       (m_data),
        .m_last       (m_last),
        .busy         (l2_busy),
        .done         (done)
    );

    assign busy = l1_busy || l2_busy;

    // l1_m_last and l1_done are not needed: the activation counter tracks
    // layer 1's output, and the inference ends with layer 2's done.
    logic unused_ok;
    assign unused_ok = l1_m_last & l1_done;

endmodule

`default_nettype wire
