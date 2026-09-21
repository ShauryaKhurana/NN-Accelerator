// =============================================================================
// relu.sv - rectified linear unit for signed two's-complement values
// =============================================================================
//     y = (x < 0) ? 0 : x
//
// Purely combinational. A two's-complement value is negative exactly when its
// sign bit is set, so no comparator is needed: the sign bit selects between x
// and zero. At gate level this is one AND gate per bit,
// y[i] = x[i] & ~x[WIDTH-1].
//
//     x[WIDTH-1] (sign) ----------+
//                                 |
//                                 v
//     x ------------------>[ sign ? 0 : x ]------> y
//
// The output is never negative, so y's sign bit is always 0. It is kept so
// that y has the same signed type and width as x and can feed signed
// arithmetic directly.
//
// Intended for the INT32 outputs of the NN layer (Phase 5), so WIDTH = 32.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module relu #(
    parameter int WIDTH = 32   // signed data width
) (
    input  logic signed [WIDTH-1:0] x,
    output logic signed [WIDTH-1:0] y
);

    assign y = x[WIDTH-1] ? '0 : x;

endmodule

`default_nettype wire
