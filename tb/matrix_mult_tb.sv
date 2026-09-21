// =============================================================================
// matrix_mult_tb.sv - self-checking testbench for matrix_mult.sv
// =============================================================================
// Each multiply: the testbench builds A and B, pulses start, waits for done,
// and compares all M*N elements of C with a reference model (a 64-bit
// integer triple loop). It also checks the documented timing: done must
// rise exactly M*N*K + 1 cycles after the start edge, as a one-cycle pulse.
//
// Sections
//   1. directed   identity patterns, zeros, all 127, all -128, -128 x 127,
//                 a checkerboard of extremes, an index pattern
//   2. protocol   start ignored while busy, reset in the middle of a run
//   3. random     N_RANDOM seeded matrix pairs (biased toward corner values),
//                 some separated by idle cycles
//   +  coverage   every behavior bin must be hit at least once
// Except where noted, each start is issued in the same cycle the previous
// done is seen, so back-to-back operation is exercised throughout.
//
// With +resultsfile=<path>, every multiply's A, B and the C read back from
// the DUT are written out for python/verify_results.py, which checks them
// against the NumPy golden model: a second, independent reference.
//
// Plusargs:   +seed=<n>  +resultsfile=<path>
//             +dumpfile=<path>  waveform run: directed section only
//             +dumpall          with +dumpfile, run and record everything
// Parameters: M, K, N (matrix dimensions), N_RANDOM
// Result:     prints "TEST PASSED" and calls $finish, or prints "TEST FAILED"
//             and calls $fatal (nonzero exit status).
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module matrix_mult_tb;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    parameter int M        = 8;
    parameter int K        = 8;
    parameter int N        = 8;
    parameter int N_RANDOM = 200;

    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int CLK_PERIOD = 10;
    localparam int MAX_ERRORS = 10;
    localparam int LATENCY    = M * N * K + 1;   // start edge -> done, per matrix_mult.sv

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic                      clk = 1'b0;
    logic                      rst;
    logic                      start;
    logic [M*K*DATA_WIDTH-1:0] a_flat;
    logic [K*N*DATA_WIDTH-1:0] b_flat;
    logic                      busy;
    logic                      done;
    logic [M*N*ACC_WIDTH-1:0]  c_flat;

    matrix_mult #(
        .M          (M),
        .K          (K),
        .N          (N),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) dut (
        .clk    (clk),
        .rst    (rst),
        .start  (start),
        .a_flat (a_flat),
        .b_flat (b_flat),
        .busy   (busy),
        .done   (done),
        .c_flat (c_flat)
    );

    always #(CLK_PERIOD / 2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Test matrices, bookkeeping, coverage
    // (everything initialized: Verilator runs with randomized initial values)
    // -------------------------------------------------------------------------
    int     a_mat [M][K];
    int     b_mat [K][N];
    longint c_exp [M][N];

    int unsigned n_checks = 0, n_errors = 0, n_mults = 0;
    int unsigned sec_checks = 0, sec_errors = 0, sec_mults = 0;
    int unsigned seed        = 1;
    logic [31:0] rng_state   = 32'h1;
    string       dumpfile    = "";
    string       resultsfile = "";
    string       run_note    = "";
    int          results_fd  = 0;
    bit          back_to_back = 1'b0;   // next start follows a done with no idle cycle

    int unsigned cov_pos = 0, cov_neg = 0, cov_zero = 0, cov_max = 0, cov_min = 0;
    int unsigned cov_back_to_back = 0, cov_idle_gap = 0, cov_ignored_start = 0, cov_reset_abort = 0;

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

    // One draw in four is a corner value, the rest are uniform INT8.
    function automatic int rand_operand();
        logic [31:0] r;
        int          value;
        r = rand32();
        if (r[1:0] != 2'b00) begin
            value = int'($signed(r[31 -: DATA_WIDTH]));
        end else begin
            case (r[4:2])
                3'd0:    value = -128;
                3'd1:    value = -127;
                3'd2:    value = -1;
                3'd3:    value = 0;
                3'd4:    value = 1;
                3'd5:    value = 126;
                default: value = 127;
            endcase
        end
        return value;
    endfunction

    task automatic fill_random();
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[i][k] = rand_operand();
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[k][j] = rand_operand();
    endtask

    task automatic fill_const(input int a_value, input int b_value);
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[i][k] = a_value;
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[k][j] = b_value;
    endtask

    // -------------------------------------------------------------------------
    // Reference model
    // -------------------------------------------------------------------------
    task automatic compute_expected();
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) begin
                c_exp[i][j] = 0;
                for (int k = 0; k < K; k++)
                    c_exp[i][j] += longint'(a_mat[i][k]) * longint'(b_mat[k][j]);
            end
    endtask

    function automatic longint dut_c(input int i, input int j);
        return longint'($signed(c_flat[(i*N + j)*ACC_WIDTH +: ACC_WIDTH]));
    endfunction

    // -------------------------------------------------------------------------
    // Driver / checker
    // -------------------------------------------------------------------------
    task automatic error(input string message);
        n_errors++;
        $display("ERROR %s", message);
        if (n_errors >= MAX_ERRORS) end_test();
    endtask

    // Called at a falling edge while the DUT is idle: present A and B, pulse
    // start for one cycle, and return at the falling edge after the start edge.
    task automatic issue_start();
        if (busy !== 1'b0) error("testbench bug: start issued while the DUT is busy");
        if (back_to_back) cov_back_to_back++;
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                a_flat[(i*K + k)*DATA_WIDTH +: DATA_WIDTH] = DATA_WIDTH'(a_mat[i][k]);
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                b_flat[(k*N + j)*DATA_WIDTH +: DATA_WIDTH] = DATA_WIDTH'(b_mat[k][j]);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        // Accepted, and the previous run's done pulse is over
        n_checks++;
        if (busy !== 1'b1 || done !== 1'b0)
            error($sformatf("after start: busy=%b done=%b, expected busy=1 done=0", busy, done));
    endtask

    // Wait (from a falling edge) until done is seen; cycles counts rising
    // edges since the start edge and continues from the value passed in.
    task automatic wait_done(inout int cycles);
        while (done !== 1'b1 && cycles <= LATENCY + 16) begin
            @(negedge clk);
            cycles++;
        end
    endtask

    task automatic idle(input int n_cycles);
        repeat (n_cycles) begin
            @(negedge clk);
            if (done !== 1'b0 || busy !== 1'b0) error("done/busy asserted while idle");
        end
        if (n_cycles > 0) begin
            back_to_back = 1'b0;
            cov_idle_gap++;
        end
    endtask

    // Compare C, check latency, record coverage, and log to the results file.
    task automatic check_result(input string name, input int cycles);
        int     bad;
        longint got;

        n_checks++;
        if (cycles != LATENCY)
            error($sformatf("%s: done after %0d cycles, expected %0d", name, cycles, LATENCY));

        bad = 0;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) begin
                got = dut_c(i, j);
                n_checks++;
                if ($isunknown(c_flat[(i*N + j)*ACC_WIDTH +: ACC_WIDTH]) || got != c_exp[i][j]) begin
                    bad++;
                    if (bad <= 3)
                        error($sformatf("%s: C[%0d][%0d] = %0d, expected %0d", name, i, j, got, c_exp[i][j]));
                end
                if (got > 0) cov_pos++;
                if (got < 0) cov_neg++;
                if (got == 0) cov_zero++;
                if (got == longint'(K) * 16384)  cov_max++;   // largest possible element
                if (got == longint'(K) * -16256) cov_min++;   // most negative possible element
            end
        if (bad > 3) error($sformatf("%s: %0d mismatching elements in total", name, bad));

        if (results_fd != 0) begin
            $fwrite(results_fd, "case %s\nA", name);
            for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) $fwrite(results_fd, " %0d", a_mat[i][k]);
            $fwrite(results_fd, "\nB");
            for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) $fwrite(results_fd, " %0d", b_mat[k][j]);
            $fwrite(results_fd, "\nC");
            for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) $fwrite(results_fd, " %0d", dut_c(i, j));
            $fwrite(results_fd, "\n");
        end

        n_mults++;
        back_to_back = 1'b1;
    endtask

    task automatic run_multiply(input string name);
        int cycles;
        compute_expected();
        issue_start();
        cycles = 0;   // rising edges after the start edge, counted at falling edges
        wait_done(cycles);
        check_result(name, cycles);
    endtask

    // Hand-computed expectation: every element of C equals value.
    task automatic expect_all(input longint value, input string name);
        int bad = 0;
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) if (dut_c(i, j) != value) bad++;
        n_checks++;
        if (bad != 0) error($sformatf("%s: %0d elements differ from the hand value %0d", name, bad, value));
        else          $display("  pass  %s: all %0d elements = %0d", name, M * N, value);
    endtask

    // -------------------------------------------------------------------------
    // Reporting
    // -------------------------------------------------------------------------
    task automatic section_begin(input string name);
        sec_checks = n_checks;
        sec_errors = n_errors;
        sec_mults  = n_mults;
        $display("");
        $display("[%s]", name);
    endtask

    task automatic section_end();
        $display("  -> %0d multiplies, %0d checks, %0d errors",
                 n_mults - sec_mults, n_checks - sec_checks, n_errors - sec_errors);
    endtask

    task automatic cover_bin(input string name, input int unsigned hits);
        $display("  %8d  %s", hits, name);
        if (hits == 0) begin
            n_errors++;
            $display("ERROR coverage hole: '%s' was never exercised", name);
        end
    endtask

    task automatic end_test();
        if (results_fd != 0) begin
            $fwrite(results_fd, "end %0d\n", n_mults);
            $fclose(results_fd);
            results_fd = 0;
        end
        $display("");
        if (n_errors == 0) begin
            $display("TEST PASSED%s: matrix_mult_tb M=%0d K=%0d N=%0d seed=%0d | %0d multiplies, %0d checks, 0 errors, latency %0d cycles",
                     run_note, M, K, N, seed, n_mults, n_checks, LATENCY);
            $finish;
        end else begin
            $display("TEST FAILED%s: matrix_mult_tb M=%0d K=%0d N=%0d seed=%0d | %0d errors in %0d checks",
                     run_note, M, K, N, seed, n_errors, n_checks);
            $fatal(1, "matrix_mult_tb failed");
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sections
    // -------------------------------------------------------------------------
    task automatic test_directed();
        section_begin("1. directed matrices");

        // Ones on the diagonal: I x B = B (rows beyond K are zero) and A x I = A
        fill_random();
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[i][k] = (i == k) ? 1 : 0;
        run_multiply("identity x random");
        fill_random();
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[k][j] = (k == j) ? 1 : 0;
        run_multiply("random x identity");

        fill_const(0, 0);
        run_multiply("zeros");
        expect_all(0, "zeros");

        fill_const(127, 127);
        run_multiply("all 127");
        expect_all(longint'(K) * 16129, "all 127");        // K * 127 * 127

        fill_const(-128, -128);
        run_multiply("all -128");
        expect_all(longint'(K) * 16384, "all -128");       // largest possible sum

        fill_const(-128, 127);
        run_multiply("-128 x 127");
        expect_all(longint'(K) * -16256, "-128 x 127");    // most negative possible sum

        // Checkerboard of the extremes: mixed signs in every dot product
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[i][k] = ((i + k) % 2 == 0) ? 127 : -128;
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[k][j] = ((k + j) % 2 == 0) ? -128 : 127;
        run_multiply("checkerboard of extremes");

        // Distinct, mixed-sign elements (up to 64 per matrix): catches swapped
        // or misrouted indices
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[i][k] = (4*(i*K + k)) % 256 - 128;
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[k][j] = 127 - (3*(k*N + j)) % 256;
        run_multiply("index pattern");

        $display("  %0d directed multiplies compared element by element; latency %0d cycles each",
                 n_mults - sec_mults, LATENCY);
        section_end();
    endtask

    task automatic test_protocol();
        int          cycles;
        logic [31:0] r;
        section_begin("2. protocol");

        // A start pulse while busy must be ignored, with different operands on
        // the buses: the result must still be the original A x B, on time.
        fill_random();
        compute_expected();
        issue_start();
        cycles = 0;
        repeat (LATENCY / 3) begin
            @(negedge clk);
            cycles++;
        end
        for (int e = 0; e < M*K; e++) begin
            r = rand32();
            a_flat[e*DATA_WIDTH +: DATA_WIDTH] = r[DATA_WIDTH-1:0];
        end
        for (int e = 0; e < K*N; e++) begin
            r = rand32();
            b_flat[e*DATA_WIDTH +: DATA_WIDTH] = r[DATA_WIDTH-1:0];
        end
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        cycles++;
        cov_ignored_start++;
        wait_done(cycles);
        check_result("start ignored while busy", cycles);
        $display("  pass  start pulse with new operands while busy was ignored");

        // Reset halfway through a run: the run is abandoned without a done
        // pulse, and the next run is correct.
        fill_random();
        compute_expected();
        issue_start();
        repeat (LATENCY / 2) @(negedge clk);
        rst = 1'b1;
        @(negedge clk);
        rst = 1'b0;
        n_checks++;
        if (busy !== 1'b0) error("busy still high after reset");
        repeat (LATENCY + 4) begin
            @(negedge clk);
            if (done !== 1'b0) error("done pulsed for a run abandoned by reset");
        end
        cov_reset_abort++;
        back_to_back = 1'b0;
        $display("  pass  reset mid-run: busy cleared, no done pulse for the abandoned run");
        fill_random();
        run_multiply("first run after reset");

        section_end();
    endtask

    task automatic test_random();
        logic [31:0] r;
        section_begin($sformatf("3. random (%0d multiplies, seed %0d)", N_RANDOM, seed));
        for (int t = 0; t < N_RANDOM; t++) begin
            fill_random();
            r = rand32();
            idle((r[1:0] == 2'b00) ? int'(r[3:2]) + 1 : 0);   // a quarter get 1-4 idle cycles
            run_multiply($sformatf("random %0d", t));
        end
        section_end();
    endtask

    task automatic report_coverage();
        section_begin("coverage (whole run)");
        cover_bin("C elements > 0",                   cov_pos);
        cover_bin("C elements < 0",                   cov_neg);
        cover_bin("C elements = 0",                   cov_zero);
        cover_bin("C element = K*16384 (largest)",    cov_max);
        cover_bin("C element = K*-16256 (most neg.)", cov_min);
        cover_bin("back-to-back starts",              cov_back_to_back);
        cover_bin("starts after idle cycles",         cov_idle_gap);
        cover_bin("start ignored while busy",         cov_ignored_start);
        cover_bin("run abandoned by reset",           cov_reset_abort);
    endtask

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    initial begin : main
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        rng_state = (seed == 0) ? 32'h1 : seed;   // xorshift32 must not start at 0

        if ($value$plusargs("resultsfile=%s", resultsfile)) begin
            results_fd = $fopen(resultsfile, "w");
            if (results_fd == 0) $fatal(1, "matrix_mult_tb: cannot open %s", resultsfile);
            $fwrite(results_fd, "# matrix_mult_tb: A, B and the C read back from the DUT, per multiply\n");
            $fwrite(results_fd, "dims %0d %0d %0d\n", M, K, N);
        end
        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, matrix_mult_tb);
        end

        $display("matrix_mult_tb: C[%0dx%0d] = A[%0dx%0d] x B[%0dx%0d], INT8 x INT8 -> INT32, latency %0d cycles",
                 M, N, M, K, K, N, LATENCY);

        rst    = 1'b1;
        start  = 1'b0;
        a_flat = '0;
        b_flat = '0;
        repeat (2) @(negedge clk);
        rst = 1'b0;

        test_directed();
        if (dumpfile != "" && !$test$plusargs("dumpall")) begin
            run_note = " (waveform run: section 1 only)";
            $display("");
            $display("waveform run: sections 2-3 skipped; add +dumpall to run and record them");
        end else begin
            test_protocol();
            test_random();
            report_coverage();
        end
        end_test();
    end

    // Watchdog: fail rather than hang if done never arrives.
    initial begin : watchdog
        repeat ((N_RANDOM + 32) * (LATENCY + 8)) @(posedge clk);
        $display("TEST FAILED: matrix_mult_tb watchdog timeout");
        $fatal(1, "matrix_mult_tb timeout");
    end

endmodule

`default_nettype wire
