// =============================================================================
// requant_tb.sv - self-checking testbench for requant.sv
// =============================================================================
// The DUT is combinational: each check drives x, waits 1 ns, and compares y
// with a 64-bit reference model, clamp(x >>> SHIFT), written from the spec
// rather than from the RTL.
//
// Sections
//   1. boundaries   0, +-1, both saturation edges and the largest/smallest
//                   inputs that still pass through, INT32 extremes
//   2. rounding     the arithmetic shift rounds toward negative infinity, so
//                   small negative inputs become -1, not 0
//   3. bit patterns walking one, positive and negative
//   4. exhaustive   every input when IN_WIDTH <= 16
//   5. random       N_RANDOM values: uniform, near zero, near the edges
//   +  coverage     saturation both ways, pass-through both signs, floor cases
//
// Plusargs:   +seed=<n>  +dumpfile=<path> (waveform run: sections 1-3 only)
// Parameters: IN_WIDTH, OUT_WIDTH, SHIFT, N_RANDOM
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module requant_tb;

    parameter int IN_WIDTH  = 32;
    parameter int OUT_WIDTH = 8;
    parameter int SHIFT     = 8;
    parameter int N_RANDOM  = 100_000;

    localparam int MAX_ERRORS = 10;

    localparam longint IN_MAX  =  (longint'(1) << (IN_WIDTH - 1)) - 1;
    localparam longint IN_MIN  = -(longint'(1) << (IN_WIDTH - 1));
    localparam longint SAT_MAX =  (longint'(1) << (OUT_WIDTH - 1)) - 1;   //  127
    localparam longint SAT_MIN = -(longint'(1) << (OUT_WIDTH - 1));       // -128
    localparam longint STEP    =  longint'(1) << SHIFT;                   // inputs per output step

    logic signed [IN_WIDTH-1:0]  x;
    logic signed [OUT_WIDTH-1:0] y;

    requant #(
        .IN_WIDTH  (IN_WIDTH),
        .OUT_WIDTH (OUT_WIDTH),
        .SHIFT     (SHIFT)
    ) dut (
        .x (x),
        .y (y)
    );

    int unsigned n_checks = 0, n_errors = 0;
    int unsigned sec_checks = 0, sec_errors = 0;
    int unsigned seed      = 1;
    logic [31:0] rng_state = 32'h1;
    string       dumpfile  = "";
    string       run_note  = "";

    int unsigned cov_sat_hi = 0, cov_sat_lo = 0, cov_pass_pos = 0, cov_pass_neg = 0;
    int unsigned cov_zero = 0, cov_floor_neg = 0;

    function automatic logic [31:0] rand32();
        rng_state = rng_state ^ (rng_state << 13);
        rng_state = rng_state ^ (rng_state >> 17);
        rng_state = rng_state ^ (rng_state << 5);
        return rng_state;
    endfunction

    // Reference: arithmetic shift (rounds toward -infinity), then clamp
    function automatic longint model(input longint value);
        longint shifted;
        shifted = value >>> SHIFT;
        if (shifted > SAT_MAX) return SAT_MAX;
        if (shifted < SAT_MIN) return SAT_MIN;
        return shifted;
    endfunction

    task automatic error(input string message);
        n_errors++;
        $display("ERROR %s", message);
        if (n_errors >= MAX_ERRORS) end_test();
    endtask

    task automatic check(input longint value, input bit verbose);
        longint expected;
        if (value < IN_MIN || value > IN_MAX)
            $fatal(1, "requant_tb bug: input %0d out of range", value);
        x = IN_WIDTH'(value);
        #1;
        expected = model(value);
        n_checks++;
        if ($isunknown(y) || longint'(y) != expected) begin
            error($sformatf("requant(%0d) = %0d, expected %0d", value, y, expected));
        end else if (verbose) begin
            $display("  pass  requant(%0d) = %0d", value, expected);
        end
        if (expected == SAT_MAX && (value >>> SHIFT) > SAT_MAX) cov_sat_hi++;
        if (expected == SAT_MIN && (value >>> SHIFT) < SAT_MIN) cov_sat_lo++;
        if (expected > 0 && expected < SAT_MAX) cov_pass_pos++;
        if (expected < 0 && expected > SAT_MIN) cov_pass_neg++;
        if (expected == 0) cov_zero++;
        if (value < 0 && (value % STEP) != 0) cov_floor_neg++;   // shift had to round down
    endtask

    task automatic section_begin(input string name);
        sec_checks = n_checks;
        sec_errors = n_errors;
        $display("");
        $display("[%s]", name);
    endtask

    task automatic section_end();
        $display("  -> %0d checks, %0d errors", n_checks - sec_checks, n_errors - sec_errors);
    endtask

    task automatic cover_bin(input string name, input int unsigned hits);
        $display("  %8d  %s", hits, name);
        if (hits == 0) begin
            n_errors++;
            $display("ERROR coverage hole: '%s' was never exercised", name);
        end
    endtask

    task automatic end_test();
        $display("");
        if (n_errors == 0) begin
            $display("TEST PASSED%s: requant_tb IN_WIDTH=%0d OUT_WIDTH=%0d SHIFT=%0d seed=%0d | %0d checks, 0 errors",
                     run_note, IN_WIDTH, OUT_WIDTH, SHIFT, seed, n_checks);
            $finish;
        end else begin
            $display("TEST FAILED%s: requant_tb IN_WIDTH=%0d OUT_WIDTH=%0d SHIFT=%0d seed=%0d | %0d errors in %0d checks",
                     run_note, IN_WIDTH, OUT_WIDTH, SHIFT, seed, n_errors, n_checks);
            $fatal(1, "requant_tb failed");
        end
    endtask

    // -------------------------------------------------------------------------
    // Sections
    // -------------------------------------------------------------------------
    task automatic test_boundaries();
        section_begin("1. boundaries");
        check(0, 1'b1);
        check(STEP, 1'b1);                      // exactly 1 after the shift
        check(-STEP, 1'b1);                     // exactly -1
        check(SAT_MAX * STEP, 1'b1);            // largest input that gives SAT_MAX
        check(SAT_MAX * STEP + (STEP - 1), 1'b1);
        if ((SAT_MAX + 1) * STEP <= IN_MAX) check((SAT_MAX + 1) * STEP, 1'b1);   // saturates
        check(SAT_MIN * STEP, 1'b1);            // exactly SAT_MIN
        if (SAT_MIN * STEP - 1 >= IN_MIN) check(SAT_MIN * STEP - 1, 1'b1);       // saturates
        check(IN_MAX, 1'b1);
        check(IN_MIN, 1'b1);
        section_end();
    endtask

    task automatic test_rounding();
        section_begin("2. rounding: the shift floors, it does not truncate toward zero");
        check(-1, 1'b1);                        // floor(-1/2^SHIFT) = -1
        check(1, 1'b1);                         // floor(+1/2^SHIFT) = 0
        if (STEP > 1) begin
            check(-(STEP - 1), 1'b1);           // still -1
            check(STEP - 1, 1'b1);              // still 0
            check(-(STEP + 1), 1'b1);           // -2
        end
        section_end();
    endtask

    task automatic test_bit_patterns();
        longint bit_i;
        section_begin("3. walking one, positive and negative");
        for (int i = 0; i < IN_WIDTH - 1; i++) begin
            bit_i = longint'(1) << i;
            check(bit_i, 1'b0);
            check(-bit_i, 1'b0);
        end
        check(IN_MIN, 1'b0);                    // the sign bit on its own
        $display("  %0d magnitude bits, both signs", IN_WIDTH - 1);
        section_end();
    endtask

    task automatic test_exhaustive();
        section_begin("4. exhaustive");
        if (IN_WIDTH > 16) begin
            $display("  skipped: 2^%0d inputs (run with IN_WIDTH<=16 to sweep them all)", IN_WIDTH);
        end else begin
            for (longint v = IN_MIN; v <= IN_MAX; v++) check(v, 1'b0);
            $display("  all %0d inputs checked", IN_MAX - IN_MIN + 1);
        end
        section_end();
    endtask

    task automatic test_random();
        logic [63:0] r;
        longint      value;
        longint      k;
        section_begin($sformatf("5. random (%0d values, seed %0d)", N_RANDOM, seed));
        for (int i = 0; i < N_RANDOM; i++) begin
            r[63:32] = rand32();
            r[31:0]  = rand32();
            k = longint'(r[9:0]);
            case (r[11:10])
                2'd0:    value = r[0] ? k : -k;                               // near zero
                2'd1:    value = (r[0] ? SAT_MAX : SAT_MIN) * STEP + k - 512; // near the edges
                2'd2:    value = (r[0] ? IN_MAX : IN_MIN) / (1 + longint'(r[14:12]));
                default: value = longint'($signed(r[63 -: IN_WIDTH]));        // uniform
            endcase
            if (value > IN_MAX) value = IN_MAX;
            if (value < IN_MIN) value = IN_MIN;
            check(value, 1'b0);
        end
        section_end();
    endtask

    task automatic report_coverage();
        section_begin("coverage");
        cover_bin("saturated at the top",        cov_sat_hi);
        cover_bin("saturated at the bottom",     cov_sat_lo);
        cover_bin("passed through, positive",    cov_pass_pos);
        cover_bin("passed through, negative",    cov_pass_neg);
        cover_bin("result 0",                    cov_zero);
        cover_bin("negative input, shift floors", cov_floor_neg);
    endtask

    initial begin : main
        if (IN_WIDTH < 4 || IN_WIDTH > 62 || OUT_WIDTH < 2 || OUT_WIDTH > IN_WIDTH)
            $fatal(1, "requant_tb: unsupported IN_WIDTH=%0d / OUT_WIDTH=%0d", IN_WIDTH, OUT_WIDTH);

        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        rng_state = (seed == 0) ? 32'h1 : seed;
        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, requant_tb);
        end

        $display("requant_tb: y = clamp(x >>> %0d) to %0d bits, inputs [%0d, %0d]",
                 SHIFT, OUT_WIDTH, IN_MIN, IN_MAX);

        test_boundaries();
        test_rounding();
        test_bit_patterns();
        if (dumpfile != "" && !$test$plusargs("dumpall")) begin
            run_note = " (waveform run: sections 1-3 only)";
            $display("");
            $display("waveform run: sections 4-5 skipped; add +dumpall to run and record them");
        end else begin
            test_exhaustive();
            test_random();
            report_coverage();
        end
        end_test();
    end

endmodule

`default_nettype wire
