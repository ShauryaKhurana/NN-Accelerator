// =============================================================================
// mac_tb.sv - self-checking testbench for mac.sv
// =============================================================================
// Scoreboard: every clock cycle the DUT's accumulator is compared with an
// independent reference model (64-bit integer arithmetic, wrapped to
// ACC_WIDTH bits exactly as the RTL documents). Directed tests also state
// their expected values by hand, so a misunderstanding shared by the RTL and
// the model cannot hide.
//
// Sections
//   1. reset               acc leaves reset at 0; reset beats en/clear
//   2. directed products   sign combinations, zeros, operand extremes
//   3. accumulation        multi-cycle sums, hold, clear, clear+load, back-to-back
//   4. exhaustive          every operand pair (65,536 for INT8)
//   5. accumulator range   largest sums that cannot wrap, then the documented wrap
//   6. random              N_RANDOM cycles of biased-random rst/clear/en/a/b
//   +  functional coverage every behavior bin must be hit at least once
//
// Timing discipline: inputs change on the falling clock edge, the DUT samples
// on the rising edge, and acc is checked on the following falling edge, so the
// testbench never races the DUT's sampling edge.
//
// Plusargs:   +seed=<n>         seed for the random section (default 1)
//             +dumpfile=<path>  waveform run: record a VCD of the directed
//                               sections 1-3 and skip the bulk sections 4-6
//             +dumpall          with +dumpfile, run and record every section
//                               (~125 MB VCD)
// Parameters: DATA_WIDTH / ACC_WIDTH can be overridden to exercise the RTL's
//             parameterization (supported: 4 <= DATA_WIDTH <= 16,
//             2*DATA_WIDTH <= ACC_WIDTH <= 62).
// Result:     prints "TEST PASSED" and calls $finish, or prints "TEST FAILED"
//             and calls $fatal (nonzero exit status).
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module mac_tb;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    parameter int DATA_WIDTH = 8;
    parameter int ACC_WIDTH  = 32;
    parameter int N_RANDOM   = 20_000;

    localparam int CLK_PERIOD = 10;
    localparam int MAX_ERRORS = 10;   // stop early instead of flooding the log

    // Operand and accumulator limits as 64-bit integers
    localparam longint OP_MAX  =  (longint'(1) << (DATA_WIDTH - 1)) - 1;  // +127 for INT8
    localparam longint OP_MIN  = -(longint'(1) << (DATA_WIDTH - 1));      // -128 for INT8
    localparam longint ACC_MIN = -(longint'(1) << (ACC_WIDTH - 1));       // -2^31 for INT32
    // Longest sum of products that can never wrap: 2^(N-2W+1) - 1 (131,071)
    localparam longint N_SAFE  =  (longint'(1) << (ACC_WIDTH - 2 * DATA_WIDTH + 1)) - 1;

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic                         clk = 1'b0;
    logic                         rst;
    logic                         en;
    logic                         clear;
    logic signed [DATA_WIDTH-1:0] a;
    logic signed [DATA_WIDTH-1:0] b;
    logic signed [ACC_WIDTH-1:0]  acc;

    mac #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) dut (
        .clk   (clk),
        .rst   (rst),
        .en    (en),
        .clear (clear),
        .a     (a),
        .b     (b),
        .acc   (acc)
    );

    always #(CLK_PERIOD / 2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Scoreboard state and bookkeeping
    // (everything initialized: Verilator runs with randomized initial values)
    // -------------------------------------------------------------------------
    longint      model_acc = 0;    // reference model's accumulator
    longint      cycle_num = 0;    // clock cycles driven so far
    int unsigned n_checks  = 0;
    int unsigned n_errors  = 0;
    int unsigned seed      = 1;
    logic [31:0] rng_state = 32'h1;
    string       dumpfile  = "";
    string       run_note  = "";   // appended to the PASS/FAIL line for partial runs

    // Per-section snapshot, for the section summary lines
    longint      sec_cycles = 0;
    int unsigned sec_checks = 0;
    int unsigned sec_errors = 0;

    // Functional coverage: how many cycles exercised each behavior
    int unsigned cov_reset = 0, cov_hold = 0, cov_accum = 0, cov_clear = 0, cov_load = 0;
    int unsigned cov_pos_pos = 0, cov_pos_neg = 0, cov_neg_pos = 0, cov_neg_neg = 0;
    int unsigned cov_zero = 0, cov_min_min = 0, cov_wrap = 0;

    // -------------------------------------------------------------------------
    // Reference model
    // -------------------------------------------------------------------------
    // Two's-complement wrap of a 64-bit value to ACC_WIDTH bits.
    function automatic longint wrap_acc(input longint value);
        logic signed [ACC_WIDTH-1:0] truncated;
        truncated = value[ACC_WIDTH-1:0];
        return longint'(truncated);
    endfunction

    // Exact (unwrapped) next accumulator value, per the behavior table in mac.sv.
    function automatic longint model_next(input longint acc_now, input bit rst_i,
                                          input bit clear_i, input bit en_i,
                                          input longint a_i, input longint b_i);
        longint sum;
        if (rst_i) return 0;
        sum = clear_i ? 0 : acc_now;
        if (en_i) sum = sum + a_i * b_i;
        return sum;
    endfunction

    // -------------------------------------------------------------------------
    // Stimulus generation
    // -------------------------------------------------------------------------
    // xorshift32 PRNG: a tiny local generator produces the same sequence on
    // every simulator, so a failing seed reproduces on Verilator and Icarus.
    function automatic logic [31:0] rand32();
        rng_state = rng_state ^ (rng_state << 13);
        rng_state = rng_state ^ (rng_state >> 17);
        rng_state = rng_state ^ (rng_state << 5);
        return rng_state;
    endfunction

    // Biased operand: one draw in four is a corner value, the rest are uniform.
    function automatic longint rand_operand();
        logic [31:0] r;
        longint      value;
        r = rand32();
        if (r[1:0] != 2'b00) begin
            value = longint'($signed(r[31 -: DATA_WIDTH]));
        end else begin
            case (r[4:2])
                3'd0:    value = OP_MIN;
                3'd1:    value = OP_MIN + 1;
                3'd2:    value = -2;
                3'd3:    value = -1;
                3'd4:    value = 0;
                3'd5:    value = 1;
                3'd6:    value = OP_MAX - 1;
                default: value = OP_MAX;
            endcase
        end
        return value;
    endfunction

    // -------------------------------------------------------------------------
    // Coverage sampling
    // -------------------------------------------------------------------------
    function automatic void sample_coverage(input bit rst_i, input bit clear_i, input bit en_i,
                                            input longint a_i, input longint b_i,
                                            input bit wrapped);
        if (rst_i)                cov_reset++;
        else if (clear_i && en_i) cov_load++;
        else if (clear_i)         cov_clear++;
        else if (en_i)            cov_accum++;
        else                      cov_hold++;

        if (!rst_i && en_i) begin   // a product entered the accumulator this cycle
            if (a_i == 0 || b_i == 0)    cov_zero++;
            else if (a_i > 0 && b_i > 0) cov_pos_pos++;
            else if (a_i > 0)            cov_pos_neg++;
            else if (b_i > 0)            cov_neg_pos++;
            else                         cov_neg_neg++;
            if (a_i == OP_MIN && b_i == OP_MIN) cov_min_min++;
        end
        if (wrapped) cov_wrap++;
    endfunction

    // -------------------------------------------------------------------------
    // Driver + checker: one clock cycle per call
    // -------------------------------------------------------------------------
    // Called at a falling edge: drives the inputs, advances the reference model,
    // waits through the rising (sampling) edge to the next falling edge, then
    // compares the DUT with the model.
    task automatic drive(input bit rst_i, input bit clear_i, input bit en_i,
                         input longint a_i, input longint b_i);
        longint exact;

        if (a_i < OP_MIN || a_i > OP_MAX || b_i < OP_MIN || b_i > OP_MAX)
            $fatal(1, "mac_tb bug: operand out of range (a=%0d b=%0d)", a_i, b_i);

        rst   = rst_i;
        clear = clear_i;
        en    = en_i;
        a     = DATA_WIDTH'(a_i);
        b     = DATA_WIDTH'(b_i);

        exact     = model_next(model_acc, rst_i, clear_i, en_i, a_i, b_i);
        model_acc = wrap_acc(exact);
        sample_coverage(rst_i, clear_i, en_i, a_i, b_i, exact != model_acc);

        @(negedge clk);
        cycle_num++;
        n_checks++;
        if ($isunknown(acc) || longint'(acc) != model_acc) begin
            n_errors++;
            $display("ERROR cycle %0d: acc=%0d expected=%0d (rst=%0b clear=%0b en=%0b a=%0d b=%0d)",
                     cycle_num, acc, model_acc, rst_i, clear_i, en_i, a_i, b_i);
            if (n_errors >= MAX_ERRORS) end_test();
        end
    endtask

    // One-cycle stimulus shorthands (rows of the behavior table in mac.sv)
    task automatic do_clear();                                 drive(1'b0, 1'b1, 1'b0, 0, 0); endtask // acc  = 0
    task automatic do_load (input longint x, input longint y); drive(1'b0, 1'b1, 1'b1, x, y); endtask // acc  = x*y
    task automatic do_accum(input longint x, input longint y); drive(1'b0, 1'b0, 1'b1, x, y); endtask // acc += x*y
    task automatic do_hold (input longint x, input longint y); drive(1'b0, 1'b0, 1'b0, x, y); endtask // x, y ignored

    // Hand-stated expectation for directed tests (independent of the model).
    task automatic expect_acc(input longint expected, input string what);
        n_checks++;
        if ($isunknown(acc) || longint'(acc) != expected) begin
            n_errors++;
            $display("ERROR %s: acc=%0d, expected %0d", what, acc, expected);
            if (n_errors >= MAX_ERRORS) end_test();
        end else begin
            $display("  pass  %s = %0d", what, expected);
        end
    endtask

    // Load x*y on its own (clear+en) and check it against a stated value.
    task automatic check_product(input longint x, input longint y, input longint expected);
        do_load(x, y);
        expect_acc(expected, $sformatf("%0d * %0d", x, y));
    endtask

    // -------------------------------------------------------------------------
    // Reporting
    // -------------------------------------------------------------------------
    task automatic section_begin(input string name);
        sec_cycles = cycle_num;
        sec_checks = n_checks;
        sec_errors = n_errors;
        $display("");
        $display("[%s]", name);
    endtask

    task automatic section_end();
        $display("  -> %0d cycles, %0d checks, %0d errors",
                 cycle_num - sec_cycles, n_checks - sec_checks, n_errors - sec_errors);
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
            $display("TEST PASSED%s: mac_tb DATA_WIDTH=%0d ACC_WIDTH=%0d seed=%0d | %0d cycles, %0d checks, 0 errors",
                     run_note, DATA_WIDTH, ACC_WIDTH, seed, cycle_num, n_checks);
            $finish;
        end else begin
            $display("TEST FAILED%s: mac_tb DATA_WIDTH=%0d ACC_WIDTH=%0d seed=%0d | %0d errors in %0d checks",
                     run_note, DATA_WIDTH, ACC_WIDTH, seed, n_errors, n_checks);
            $fatal(1, "mac_tb failed");
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sections
    // -------------------------------------------------------------------------
    task automatic test_reset();
        section_begin("1. reset");
        // Reset held for two cycles while en/clear and operands are active:
        // reset must win. Verilator starts every flop at a random value and
        // Icarus starts them at X, so this also proves reset initializes acc.
        drive(1'b1, 1'b0, 1'b1, OP_MAX, OP_MAX);
        drive(1'b1, 1'b1, 1'b1, OP_MIN, OP_MIN);
        expect_acc(0, "acc after reset (en and clear also asserted)");
        // Reset in the middle of a sum
        do_load(5, 6);
        do_accum(7, 3);
        expect_acc(51, "5*6 + 7*3");
        drive(1'b1, 1'b0, 1'b1, 7, 7);
        expect_acc(0, "reset while accumulating");
        do_accum(2, 3);
        expect_acc(6, "accumulation restarts from 0 after reset");
        section_end();
    endtask

    task automatic test_products();
        section_begin("2. directed products (each loaded alone with clear+en)");
        check_product( 3,  5,  15);                  // positive * positive
        check_product(-7,  6, -42);                  // negative * positive
        check_product( 4, -3, -12);                  // positive * negative
        check_product(-5, -6,  30);                  // negative * negative
        check_product( 1, -1,  -1);
        check_product(-1, -1,   1);
        check_product( 0,  0,   0);                  // zeros
        check_product( 0,      OP_MAX, 0);
        check_product( OP_MIN, 0,      0);
        check_product(OP_MAX, OP_MAX, OP_MAX * OP_MAX);  //  127 *  127 =  16129
        check_product(OP_MIN, OP_MIN, OP_MIN * OP_MIN);  // -128 * -128 = +16384: needs all 16 bits
        check_product(OP_MIN, OP_MAX, OP_MIN * OP_MAX);  // -128 *  127 = -16256
        check_product(OP_MAX, OP_MIN, OP_MAX * OP_MIN);
        check_product(OP_MIN,  1,     OP_MIN);
        check_product(OP_MIN, -1,     -OP_MIN);          // +128: not an INT8, but a valid product
        section_end();
    endtask

    task automatic test_accumulation();
        section_begin("3. accumulation and control");
        // Multi-cycle sum with mixed signs:
        //   [1, -2, 3, 4, -5, 6, 7] . [7, 6, -5, 4, 3, -2, 1]
        //   = 7 - 12 - 15 + 16 - 15 - 12 + 7 = -24
        do_clear();
        expect_acc(0, "clear");
        do_accum( 1,  7); do_accum(-2,  6); do_accum( 3, -5); do_accum(4, 4);
        do_accum(-5,  3); do_accum( 6, -2); do_accum( 7,  1);
        expect_acc(-24, "7-term dot product");

        // en low: operands must be ignored and the sum held
        do_hold(OP_MAX, OP_MAX); do_hold(OP_MIN, OP_MAX); do_hold(-3, 2);
        expect_acc(-24, "held for 3 cycles with en=0 and changing operands");
        do_accum(-8, -3);
        expect_acc(0, "accumulation resumes: -24 + (-8 * -3)");

        // clear alone zeroes the sum; clear with en loads the product
        do_accum(6, 7);
        do_clear();
        expect_acc(0, "clear with en=0 discards the sum");
        do_accum(5, 5); do_accum(2, 3);
        expect_acc(31, "5*5 + 2*3");
        do_load(-4, 7);
        expect_acc(-28, "clear with en=1 discards 31 and loads -4*7");

        // Back-to-back dot products with no idle cycle between them
        do_load(2, 5); do_accum(3, 6); do_accum(4, 7);
        expect_acc(56, "dot product A: [2,3,4].[5,6,7]");
        do_load(-1, 7); do_accum(-2, 7); do_accum(-3, 7);
        expect_acc(-42, "dot product B: [-1,-2,-3].[7,7,7], started with no bubble");
        section_end();
    endtask

    task automatic test_exhaustive();
        section_begin("4. exhaustive operand sweep");
        if (DATA_WIDTH > 8) begin
            $display("  skipped: 2^%0d operand pairs", 2 * DATA_WIDTH);
        end else begin
            for (longint x = OP_MIN; x <= OP_MAX; x++)
                for (longint y = OP_MIN; y <= OP_MAX; y++)
                    do_load(x, y);
            $display("  all %0d operand pairs loaded and checked against the model",
                     (OP_MAX - OP_MIN + 1) * (OP_MAX - OP_MIN + 1));
        end
        section_end();
    endtask

    task automatic test_range();
        section_begin("5. accumulator range and wrap");
        if (N_SAFE > 200_000) begin   // keeps the run well inside the watchdog
            $display("  skipped: %0d-term boundary is too long to simulate", N_SAFE);
        end else begin
            // Largest positive sum that cannot wrap: N_SAFE x (MIN*MIN)
            do_load(OP_MIN, OP_MIN);
            repeat (int'(N_SAFE - 1)) do_accum(OP_MIN, OP_MIN);
            expect_acc(N_SAFE * OP_MIN * OP_MIN,
                       $sformatf("%0d x (%0d * %0d), largest positive sum", N_SAFE, OP_MIN, OP_MIN));
            // One more term crosses +2^(N-1) and wraps to the most negative value
            do_accum(OP_MIN, OP_MIN);
            expect_acc(ACC_MIN, "one more term wraps to -2^(ACC_WIDTH-1), as documented");
            // Most negative sum of the same length: N_SAFE x (MIN*MAX)
            do_load(OP_MIN, OP_MAX);
            repeat (int'(N_SAFE - 1)) do_accum(OP_MIN, OP_MAX);
            expect_acc(N_SAFE * OP_MIN * OP_MAX,
                       $sformatf("%0d x (%0d * %0d), most negative sum", N_SAFE, OP_MIN, OP_MAX));
        end
        section_end();
    endtask

    task automatic test_random();
        logic [31:0] r;
        longint      x, y;
        section_begin($sformatf("6. constrained random (%0d cycles, seed %0d)", N_RANDOM, seed));
        for (int i = 0; i < N_RANDOM; i++) begin
            r = rand32();
            x = rand_operand();
            y = rand_operand();
            drive(r[7:0] == 8'd0,     // rst:   ~1/256 of cycles
                  r[11:8] == 4'd0,    // clear: ~1/16
                  r[13:12] != 2'd0,   // en:    ~3/4
                  x, y);
        end
        section_end();
    endtask

    task automatic report_coverage();
        section_begin("functional coverage (cycles exercising each behavior, whole run)");
        cover_bin("reset",                      cov_reset);
        cover_bin("hold (en=0)",                cov_hold);
        cover_bin("accumulate (en=1)",          cov_accum);
        cover_bin("clear (clear=1, en=0)",      cov_clear);
        cover_bin("clear+load (clear=1, en=1)", cov_load);
        cover_bin("product: pos * pos",         cov_pos_pos);
        cover_bin("product: pos * neg",         cov_pos_neg);
        cover_bin("product: neg * pos",         cov_neg_pos);
        cover_bin("product: neg * neg",         cov_neg_neg);
        cover_bin("product: zero operand",      cov_zero);
        cover_bin("product: MIN * MIN",         cov_min_min);
        cover_bin("accumulator wrap",           cov_wrap);
    endtask

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    initial begin : main
        if (DATA_WIDTH < 4 || DATA_WIDTH > 16 || ACC_WIDTH < 2 * DATA_WIDTH || ACC_WIDTH > 62)
            $fatal(1, "mac_tb: unsupported DATA_WIDTH=%0d / ACC_WIDTH=%0d", DATA_WIDTH, ACC_WIDTH);

        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        rng_state = (seed == 0) ? 32'h1 : seed;   // xorshift32 must not start at 0

        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, mac_tb);
        end

        $display("mac_tb: DATA_WIDTH=%0d ACC_WIDTH=%0d  operands [%0d, %0d]  no-wrap bound %0d terms",
                 DATA_WIDTH, ACC_WIDTH, OP_MIN, OP_MAX, N_SAFE);

        test_reset();
        test_products();
        test_accumulation();
        // A waveform run ends after the directed sections: recording the bulk
        // sections (~350k cycles) makes a ~125 MB VCD. ($dumpoff cannot limit
        // it portably: Verilator 5 accepts $dumpoff but ignores it.)
        if (dumpfile != "" && !$test$plusargs("dumpall")) begin
            run_note = " (waveform run: sections 1-3 only)";
            $display("");
            $display("waveform run: sections 4-6 skipped; add +dumpall to run and record them");
        end else begin
            test_exhaustive();
            test_range();
            test_random();
            report_coverage();
        end
        end_test();
    end

    // Watchdog: fail rather than hang if the sequence ever stalls.
    initial begin : watchdog
        repeat (1_000_000 + N_RANDOM) @(posedge clk);
        $display("TEST FAILED: mac_tb watchdog timeout");
        $fatal(1, "mac_tb timeout");
    end

endmodule

`default_nettype wire
