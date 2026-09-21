// =============================================================================
// mac.sv - signed multiply-accumulate unit
// =============================================================================
// One signed multiply-accumulate per clock:
//
//     acc <= acc + a * b        (INT8 x INT8 -> INT16 product -> INT32 sum)
//
// Datapath (W = DATA_WIDTH, N = ACC_WIDTH):
//
//   a --+   product     sign-extend       addend
//       +-->[ x ]------>[ 2W -> N ]--->[ en ? p : 0 ]---+
//   b --+   2W bits                                     |
//                                                       v
//           +-->[ clear ? 0 : acc ]--- feedback --->[ + ]-->[ acc reg ]--+--> acc
//           |                                                            |
//           +------------------------------------------------------------+
//
// Control (a single register; reset has priority):
//
//     rst clear en | acc after the next rising edge
//     -------------+----------------------------------------------
//      1    -    - | 0            synchronous reset
//      0    1    1 | a*b          start a new sum with this product
//      0    1    0 | 0            discard the running sum
//      0    0    1 | acc + a*b    accumulate
//      0    0    0 | acc          hold (a and b are ignored)
//
// "clear with en" loads the product instead of zero, so back-to-back dot
// products need no idle cycle between them. This is the same feedback-mux
// structure an FPGA DSP slice uses for its accumulator.
//
// Numeric range:
//   * A W-bit x W-bit signed product always fits in 2W bits. The extreme case
//     is (-2^(W-1))^2 = +2^(2W-2), e.g. -128 * -128 = +16384, so the multiply
//     is exact.
//   * The accumulator does not saturate; it wraps modulo 2^N (two's
//     complement). No sum of up to 2^(N-2W+1) - 1 products can wrap: for
//     INT8/INT32 that is 131,071 terms, far longer than any dot product in
//     this project. The testbench checks both sides of this boundary.
//
// Timing: combinational multiply and add, one register. The result of an
// enabled cycle appears on acc one clock later.
// Reset: synchronous, active-high. FPGA DSP-slice registers reset
// synchronously, so this style lets the MAC map onto a hard DSP block later.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module mac #(
    parameter int DATA_WIDTH = 8,   // signed operand width
    parameter int ACC_WIDTH  = 32   // signed accumulator width, >= 2*DATA_WIDTH
) (
    input  logic                         clk,
    input  logic                         rst,    // synchronous, active-high
    input  logic                         en,     // a*b is valid this cycle: accumulate it
    input  logic                         clear,  // start a new sum (see table above)
    input  logic signed [DATA_WIDTH-1:0] a,
    input  logic signed [DATA_WIDTH-1:0] b,
    output logic signed [ACC_WIDTH-1:0]  acc
);

    localparam int PROD_WIDTH = 2 * DATA_WIDTH;

    // A narrower accumulator could not hold even a single product.
    // (Plain-string message: Icarus rejects format arguments in elaboration tasks.)
    if (ACC_WIDTH < PROD_WIDTH) begin : g_acc_width_check
        $error("mac: ACC_WIDTH must be >= 2*DATA_WIDTH");
    end

    logic signed [PROD_WIDTH-1:0] product;      // exact W x W product
    logic signed [ACC_WIDTH-1:0]  product_ext;  // product sign-extended to accumulator width
    logic signed [ACC_WIDTH-1:0]  addend;       // product, or 0 when en is low
    logic signed [ACC_WIDTH-1:0]  feedback;     // running sum, or 0 when clear is high

    // Both operands are widened to the full product width before multiplying.
    // A size cast keeps its operand's signedness, so these casts sign-extend,
    // and a multiply of two signed operands is a signed multiply.
    assign product     = PROD_WIDTH'(a) * PROD_WIDTH'(b);
    assign product_ext = ACC_WIDTH'(product);

    assign addend   = en    ? product_ext : '0;
    assign feedback = clear ? '0 : acc;

    always_ff @(posedge clk) begin
        if (rst) acc <= '0;
        else     acc <= feedback + addend;
    end

endmodule

`default_nettype wire
