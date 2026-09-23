// =============================================================================
// matmul_pkg.sv - types shared by the matrix-multiply controller and its
// testbenches
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

package matmul_pkg;

    // Controller states (see rtl/matmul_ctrl.sv for the transition table).
    // Encodings are fixed so the state is easy to read as a number in a VCD.
    typedef enum logic [2:0] {
        S_IDLE      = 3'd0,
        S_LOAD      = 3'd1,
        S_COMPUTE   = 3'd2,
        S_WRITEBACK = 3'd3,
        S_DONE      = 3'd4
    } matmul_state_t;

endpackage

`default_nettype wire
