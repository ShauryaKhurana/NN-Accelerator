// =============================================================================
// requant.sv - INT32 activation to INT8: arithmetic right shift, then saturate
// =============================================================================
//     y = clamp(x >>> SHIFT, -2^(OUT_WIDTH-1), 2^(OUT_WIDTH-1) - 1)
//
// Between two INT8 layers the activations have to come back down to INT8: a
// layer accumulates in INT32, but the next layer's multiplier takes INT8. This
// is the usual fixed-point rescaling, with the scale factor restricted to a
// power of two so it costs no multiplier:
//
//   x (INT32) --->[ >>> SHIFT ]--->[ saturate to OUT_WIDTH ]---> y (INT8)
//                  arithmetic          clamp, do not wrap
//
// The shift is arithmetic, so it rounds toward negative infinity (floor), the
// same as an integer shift in Python or NumPy. Saturation clamps instead of
// wrapping, so a large activation becomes the largest representable value
// rather than changing sign.
//
// Purely combinational; a constant shift is just wiring, and saturation is a
// pair of comparisons and a mux.
//
// Choosing SHIFT: with K INT8 inputs the sums reach about K * 2^14, so the
// shift sets how much of that range survives. The default 8 keeps typical
// activations of a 16-input layer around 70 and saturates only the largest
// few percent. It is a parameter so it can be swept.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module requant #(
    parameter int IN_WIDTH  = 32,   // signed accumulator width
    parameter int OUT_WIDTH = 8,    // signed activation width
    parameter int SHIFT     = 8     // arithmetic right shift applied first
) (
    input  logic signed [IN_WIDTH-1:0]  x,
    output logic signed [OUT_WIDTH-1:0] y
);

    // (Plain-string messages: Icarus rejects format arguments in elaboration tasks.)
    if (OUT_WIDTH > IN_WIDTH) begin : g_width_check
        $error("requant: OUT_WIDTH must be <= IN_WIDTH");
    end
    if (SHIFT < 0) begin : g_shift_check
        $error("requant: SHIFT must be >= 0");
    end

    localparam int SAT_MAX_I = (1 << (OUT_WIDTH - 1)) - 1;    //  127 for INT8
    localparam int SAT_MIN_I = -(1 << (OUT_WIDTH - 1));       // -128 for INT8

    localparam logic signed [IN_WIDTH-1:0] SAT_MAX = IN_WIDTH'(SAT_MAX_I);
    localparam logic signed [IN_WIDTH-1:0] SAT_MIN = IN_WIDTH'(SAT_MIN_I);

    logic signed [IN_WIDTH-1:0] shifted;

    assign shifted = x >>> SHIFT;   // x is signed, so this is an arithmetic shift

    always_comb begin
        if (shifted > SAT_MAX)      y = OUT_WIDTH'(SAT_MAX_I);
        else if (shifted < SAT_MIN) y = OUT_WIDTH'(SAT_MIN_I);
        else                        y = OUT_WIDTH'(shifted);
    end

endmodule

`default_nettype wire
