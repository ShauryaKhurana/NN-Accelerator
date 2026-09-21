// =============================================================================
// relu_tb.sv - self-checking testbench for relu.sv
// =============================================================================
// The DUT is combinational: each check drives x, waits 1 ns for y to settle,
// and compares y with the reference model (64-bit integer math:
// y = x < 0 ? 0 : x). The model uses a signed comparison, while the RTL uses
// the sign bit, so the two are written independently.
//
// Sections
//   1. boundaries     0, +-1, +-2, +-100, MAX, MAX-1, MIN, MIN+1
//   2. bit patterns   walking-one and walking-zero, each with the sign bit
//                     clear (must pass through) and set (must become 0)
//   3. exhaustive     every input value, when WIDTH <= 16 (65,536 values)
//   4. random         N_RANDOM values: uniform, near zero, near the extremes
//   +  coverage       every input bit seen at 0 and 1, every output magnitude
//                     bit seen at 1, output sign bit never set
//
// Plusargs:   +seed=<n>         seed for the random section (default 1)
//             +dumpfile=<path>  waveform run: record sections 1-2 only
//             +dumpall          with +dumpfile, run and record every section
// Parameters: WIDTH (supported: 8..62), N_RANDOM
// Result:     prints "TEST PASSED" and calls $finish, or prints "TEST FAILED"
//             and calls $fatal (nonzero exit status).
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module relu_tb;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    parameter int WIDTH    = 32;
    parameter int N_RANDOM = 100_000;

    localparam int MAX_ERRORS = 10;

    localparam longint X_MAX =  (longint'(1) << (WIDTH - 1)) - 1;  // +2^31 - 1 for WIDTH=32
    localparam longint X_MIN = -(longint'(1) << (WIDTH - 1));      // -2^31

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic signed [WIDTH-1:0] x;
    logic signed [WIDTH-1:0] y;

    relu #(
        .WIDTH (WIDTH)
    ) dut (
        .x (x),
        .y (y)
    );

    // -------------------------------------------------------------------------
    // Bookkeeping (everything initialized: Verilator randomizes initial values)
    // -------------------------------------------------------------------------
    int unsigned n_checks  = 0;
    int unsigned n_errors  = 0;
    int unsigned seed      = 1;
    logic [31:0] rng_state = 32'h1;
    string       dumpfile  = "";
    string       run_note  = "";   // appended to the PASS/FAIL line for partial runs

    int unsigned sec_checks = 0;
    int unsigned sec_errors = 0;

    // Coverage
    int unsigned      cov_neg = 0, cov_zero = 0, cov_pos = 0, cov_min = 0, cov_max = 0;
    logic [WIDTH-1:0] in_seen_1  = '0;   // input bits observed at 1
    logic [WIDTH-1:0] in_seen_0  = '0;   // input bits observed at 0
    logic [WIDTH-1:0] out_seen_1 = '0;   // output bits observed at 1

    // -------------------------------------------------------------------------
    // Stimulus helpers
    // -------------------------------------------------------------------------
    // xorshift32 PRNG: identical sequence on every simulator.
    function automatic logic [31:0] rand32();
        rng_state = rng_state ^ (rng_state << 13);
        rng_state = rng_state ^ (rng_state >> 17);
        rng_state = rng_state ^ (rng_state << 5);
        return rng_state;
    endfunction

    // Biased input: half uniform over the full range, a quarter within 255 of
    // zero, a quarter within 255 of MAX or MIN (where sign handling matters).
    function automatic longint rand_value();
        logic [63:0] r;
        longint      k;
        longint      value;
        r[63:32] = rand32();
        r[31:0]  = rand32();
        k = longint'(r[7:0]);
        case (r[9:8])
            2'd0:    value = r[10] ? k : -k;
            2'd1:    value = r[10] ? X_MAX - k : X_MIN + k;
            default: value = longint'($signed(r[63 -: WIDTH]));
        endcase
        return value;
    endfunction

    // -------------------------------------------------------------------------
    // Driver + checker
    // -------------------------------------------------------------------------
    task automatic check(input longint value, input bit verbose);
        longint expected;

        if (value < X_MIN || value > X_MAX)
            $fatal(1, "relu_tb bug: input %0d out of range", value);

        x = WIDTH'(value);
        #1;
        expected = (value < 0) ? 0 : value;

        n_checks++;
        if (value < 0)      cov_neg++;
        if (value == 0)     cov_zero++;
        if (value > 0)      cov_pos++;
        if (value == X_MIN) cov_min++;
        if (value == X_MAX) cov_max++;
        in_seen_1  = in_seen_1  | x;
        in_seen_0  = in_seen_0  | ~x;
        out_seen_1 = out_seen_1 | y;

        if ($isunknown(y) || longint'(y) != expected) begin
            n_errors++;
            $display("ERROR relu(%0d) = %0d, expected %0d", value, y, expected);
            if (n_errors >= MAX_ERRORS) end_test();
        end else if (verbose) begin
            $display("  pass  relu(%0d) = %0d", value, expected);
        end
    endtask

    // -------------------------------------------------------------------------
    // Reporting
    // -------------------------------------------------------------------------
    task automatic section_begin(input string name);
        sec_checks = n_checks;
        sec_errors = n_errors;
        $display("");
        $display("[%s]", name);
    endtask

    task automatic section_end();
        $display("  -> %0d checks, %0d errors", n_checks - sec_checks, n_errors - sec_errors);
    endtask

    task automatic cover_bin(input string name, input bit hit, input string detail);
        $display("  %s  %s  %s", hit ? "hit " : "MISS", name, detail);
        if (!hit) begin
            n_errors++;
            $display("ERROR coverage hole: %s", name);
        end
    endtask

    task automatic end_test();
        $display("");
        if (n_errors == 0) begin
            $display("TEST PASSED%s: relu_tb WIDTH=%0d seed=%0d | %0d checks, 0 errors",
                     run_note, WIDTH, seed, n_checks);
            $finish;
        end else begin
            $display("TEST FAILED%s: relu_tb WIDTH=%0d seed=%0d | %0d errors in %0d checks",
                     run_note, WIDTH, seed, n_errors, n_checks);
            $fatal(1, "relu_tb failed");
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sections
    // -------------------------------------------------------------------------
    task automatic test_boundaries();
        section_begin("1. boundaries");
        check(0,         1'b1);
        check(1,         1'b1);
        check(-1,        1'b1);
        check(2,         1'b1);
        check(-2,        1'b1);
        check(100,       1'b1);
        check(-100,      1'b1);
        check(X_MAX,     1'b1);   // largest positive passes through unchanged
        check(X_MAX - 1, 1'b1);
        check(X_MIN,     1'b1);   // most negative (sign bit only) -> 0
        check(X_MIN + 1, 1'b1);
        section_end();
    endtask

    // Every magnitude bit must pass through when x >= 0 and be zeroed when
    // x < 0; walking a one (and a zero) across the word shows this bit by bit.
    task automatic test_bit_patterns();
        longint bit_i;
        section_begin("2. walking-one / walking-zero patterns, sign clear and set");
        for (int i = 0; i < WIDTH - 1; i++) begin
            bit_i = longint'(1) << i;
            check(bit_i,          1'b0);   // 0...010...0  positive: passes
            check(X_MIN + bit_i,  1'b0);   // 1...010...0  negative: 0
            check(X_MAX - bit_i,  1'b0);   // 01..101..1   positive: passes
            check(-1 - bit_i,     1'b0);   // 11..101..1   negative: 0
        end
        $display("  %0d magnitude bits x 4 patterns", WIDTH - 1);
        section_end();
    endtask

    task automatic test_exhaustive();
        section_begin("3. exhaustive");
        if (WIDTH > 16) begin
            $display("  skipped: 2^%0d values (run with WIDTH<=16 for an exhaustive sweep)", WIDTH);
        end else begin
            for (longint v = X_MIN; v <= X_MAX; v++) check(v, 1'b0);
            $display("  all %0d input values checked", X_MAX - X_MIN + 1);
        end
        section_end();
    endtask

    task automatic test_random();
        section_begin($sformatf("4. random (%0d values, seed %0d)", N_RANDOM, seed));
        for (int i = 0; i < N_RANDOM; i++) check(rand_value(), 1'b0);
        section_end();
    endtask

    task automatic report_coverage();
        section_begin("coverage");
        cover_bin("negative inputs",  cov_neg > 0, $sformatf("(%0d)", cov_neg));
        cover_bin("zero input",       cov_zero > 0, $sformatf("(%0d)", cov_zero));
        cover_bin("positive inputs",  cov_pos > 0, $sformatf("(%0d)", cov_pos));
        cover_bin("MIN input",        cov_min > 0, $sformatf("(%0d)", cov_min));
        cover_bin("MAX input",        cov_max > 0, $sformatf("(%0d)", cov_max));
        cover_bin("every input bit seen at 0 and 1", (&in_seen_1) && (&in_seen_0), "");
        cover_bin("every output magnitude bit seen at 1", &out_seen_1[WIDTH-2:0], "");
        cover_bin("output sign bit never set", !out_seen_1[WIDTH-1], "");
    endtask

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    initial begin : main
        if (WIDTH < 8 || WIDTH > 62)
            $fatal(1, "relu_tb: unsupported WIDTH=%0d", WIDTH);

        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        rng_state = (seed == 0) ? 32'h1 : seed;   // xorshift32 must not start at 0

        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, relu_tb);
        end

        $display("relu_tb: WIDTH=%0d  range [%0d, %0d]", WIDTH, X_MIN, X_MAX);

        test_boundaries();
        test_bit_patterns();
        // Waveform runs stop after the directed sections (see mac_tb.sv).
        if (dumpfile != "" && !$test$plusargs("dumpall")) begin
            run_note = " (waveform run: sections 1-2 only)";
            $display("");
            $display("waveform run: sections 3-4 skipped; add +dumpall to run and record them");
        end else begin
            test_exhaustive();
            test_random();
            report_coverage();
        end
        end_test();
    end

endmodule

`default_nettype wire
