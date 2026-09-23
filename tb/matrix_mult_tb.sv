// =============================================================================
// matrix_mult_tb.sv - testbench for the streaming matrix multiplier
// =============================================================================
// Streams A and B into the DUT and takes C out, with stalls on either side,
// and checks:
//   * values: every C element, in order, against a 64-bit reference model,
//     plus the position of m_last
//   * protocol (monitor, every rising edge): while m_valid is high and m_ready
//     low, m_valid, m_data and m_last must not change; each job takes exactly
//     M*K + K*N input beats and M*N output beats; done is a one-cycle pulse
//     after the last output beat
//   * timing: busy lasts exactly M*K + K*N + M*N*(K+1) + 1 cycles per job,
//     plus one per stall cycle (s_valid low in LOAD, m_ready low in
//     WRITEBACK); back-to-back jobs without stalls finish every 706 cycles
//     (8x8x8)
// Every completed job's A, B and received C go to the results file, which
// python/verify_results.py checks against the NumPy golden model.
//
// Sections
//   1. directed    identity patterns, zeros, all 127, all -128, -128 x 127,
//                  a checkerboard of extremes, an index pattern; no stalls,
//                  back to back, exact cycle counts
//   2. protocol    input gaps, output backpressure, both; a source that
//                  queues the next job while the current one computes;
//                  reset during LOAD, COMPUTE and WRITEBACK
//   3. random      N_RANDOM jobs with random matrices and stall rates
//   +  coverage    every bin must be hit at least once
// The driver follows the valid/ready rules: once s_valid is raised it stays
// high, with s_data unchanged, until the beat is accepted.
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
    parameter int NUM_MACS = 1;
    parameter int N_RANDOM = 200;

    localparam int DATA_WIDTH  = 8;
    localparam int ACC_WIDTH   = 32;
    localparam int CLK_PERIOD  = 10;
    localparam int MAX_ERRORS  = 10;
    localparam int LOAD_BEATS  = M*K + K*N;
    localparam int GROUPS      = (N + NUM_MACS - 1) / NUM_MACS;
    // load, then per row: one group of NUM_MACS columns every K cycles, then
    // one output beat per column; plus DONE
    localparam int BUSY_CYCLES = LOAD_BEATS + M*GROUPS*K + M*N + 1;
    localparam int PERIOD      = BUSY_CYCLES + 1;                // plus the IDLE cycle

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic                         clk = 1'b0;
    logic                         rst;
    logic                         s_valid;
    logic                         s_ready;
    logic signed [DATA_WIDTH-1:0] s_data;
    logic                         m_valid;
    logic                         m_ready;
    logic signed [ACC_WIDTH-1:0]  m_data;
    logic                         m_last;
    logic                         busy;
    logic                         done;

    matrix_mult #(
        .M          (M),
        .K          (K),
        .N          (N),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .NUM_MACS   (NUM_MACS)
    ) dut (
        .clk     (clk),
        .rst     (rst),
        .s_valid (s_valid),
        .s_ready (s_ready),
        .s_data  (s_data),
        .m_valid (m_valid),
        .m_ready (m_ready),
        .m_data  (m_data),
        .m_last  (m_last),
        .busy    (busy),
        .done    (done)
    );

    always #(CLK_PERIOD / 2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Job data: two slots, so a second job can be queued behind the first
    // (everything initialized: Verilator runs with randomized initial values)
    // -------------------------------------------------------------------------
    int     a_mat [2][M][K];
    int     b_mat [2][K][N];
    longint c_exp [2][M][N];
    longint c_got [2][M][N];

    int unsigned n_checks = 0, n_errors = 0, n_jobs = 0;
    int unsigned sec_checks = 0, sec_errors = 0, sec_jobs = 0;
    int unsigned seed        = 1;
    // Separate random streams for the main sequence, the source and the sink.
    // The source and sink run concurrently, and the order in which simulators
    // resume concurrent processes is unspecified; separate streams keep every
    // draw independent of that order, so a seed gives identical stimulus in
    // both simulators.
    logic [31:0] rng_state   = 32'h1;   // matrices, stall rates, idle gaps
    logic [31:0] rng_src     = 32'h1;   // source: gaps before input beats
    logic [31:0] rng_sink    = 32'h1;   // sink: m_ready stalls
    string       dumpfile    = "";
    string       resultsfile = "";
    string       run_note    = "";
    int          results_fd  = 0;

    // Monitor state
    longint      cycle = 0;
    int unsigned job_in = 0, job_out = 0, job_lasts = 0, job_busy = 0, job_stalls = 0;
    int unsigned last_busy = 0, idle_run = 0;
    longint      last_done_cycle = -1, last_period = 0;
    bit          prev_stalled = 1'b0, prev_done = 1'b0, prev_busy = 1'b0;
    logic signed [ACC_WIDTH-1:0] prev_m_data = '0;
    logic                        prev_m_last = 1'b0;

    // Coverage
    int unsigned cov_pos = 0, cov_neg = 0, cov_zero = 0, cov_max = 0, cov_min = 0;
    int unsigned cov_gap = 0, cov_backpressure = 0, cov_source_wait = 0, cov_back_to_back = 0;
    int unsigned cov_reset_load = 0, cov_reset_compute = 0, cov_reset_writeback = 0;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    // xorshift32: identical sequence on every simulator
    function automatic logic [31:0] xorshift32(input logic [31:0] x);
        x = x ^ (x << 13);
        x = x ^ (x >> 17);
        x = x ^ (x << 5);
        return x;
    endfunction

    function automatic logic [31:0] rand32();
        rng_state = xorshift32(rng_state);
        return rng_state;
    endfunction

    function automatic bit src_chance(input int percent);
        rng_src = xorshift32(rng_src);
        return (rng_src % 100) < percent;
    endfunction

    function automatic bit sink_chance(input int percent);
        rng_sink = xorshift32(rng_sink);
        return (rng_sink % 100) < percent;
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

    function automatic int stall_rate(input logic [1:0] sel);   // percent
        case (sel)
            2'd0:    return 0;
            2'd1:    return 10;
            2'd2:    return 50;
            default: return 90;
        endcase
    endfunction

    task automatic error(input string message);
        n_errors++;
        $display("ERROR cycle %0d: %s", cycle, message);
        if (n_errors >= MAX_ERRORS) end_test();
    endtask

    task automatic fill_random(input int slot);
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[slot][i][k] = rand_operand();
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[slot][k][j] = rand_operand();
    endtask

    task automatic fill_const(input int a_value, input int b_value);
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[0][i][k] = a_value;
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[0][k][j] = b_value;
    endtask

    task automatic compute_expected(input int slot);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) begin
                c_exp[slot][i][j] = 0;
                for (int k = 0; k < K; k++)
                    c_exp[slot][i][j] += longint'(a_mat[slot][i][k]) * longint'(b_mat[slot][k][j]);
            end
    endtask

    // -------------------------------------------------------------------------
    // Monitor: protocol rules, beat counts and cycle accounting per job
    // (signals sampled at the rising edge, before the DUT's registers update)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        cycle++;
        if (rst) begin
            job_in = 0; job_out = 0; job_lasts = 0; job_busy = 0; job_stalls = 0;
            prev_stalled = 1'b0;
            prev_done    = 1'b0;
            prev_busy    = 1'b0;
            idle_run     = 0;
        end else begin
            // An offered output beat must hold until it is taken
            if (prev_stalled && (m_valid !== 1'b1 || m_data !== prev_m_data || m_last !== prev_m_last))
                error("output changed while m_valid was high and m_ready low");
            prev_stalled = m_valid && !m_ready;
            prev_m_data  = m_data;
            prev_m_last  = m_last;

            if (s_valid && s_ready) job_in++;
            if (m_valid && m_ready) begin
                job_out++;
                if (m_last) job_lasts++;
            end
            if (busy) job_busy++;
            if (busy && s_ready && !s_valid) begin job_stalls++; cov_gap++;          end
            if (m_valid && !m_ready)         begin job_stalls++; cov_backpressure++; end
            if (busy && s_valid && !s_ready) cov_source_wait++;

            if (!busy) idle_run++;
            if (busy && !prev_busy) begin          // a job started (IDLE -> LOAD)
                if (idle_run == 1 && last_done_cycle >= 0) cov_back_to_back++;
                idle_run = 0;
            end
            prev_busy = busy;

            if (done) begin
                n_checks++;
                if (prev_done) error("done high for more than one cycle");
                if (job_in != LOAD_BEATS || job_out != M*N || job_lasts != 1)
                    error($sformatf("job had %0d input beats, %0d output beats, %0d m_last (expected %0d, %0d, 1)",
                                    job_in, job_out, job_lasts, LOAD_BEATS, M*N));
                if (job_busy != BUSY_CYCLES + job_stalls)
                    error($sformatf("job busy for %0d cycles, expected %0d + %0d stalls",
                                    job_busy, BUSY_CYCLES, job_stalls));
                last_busy   = job_busy;
                last_period = (last_done_cycle >= 0) ? cycle - last_done_cycle : 0;
                last_done_cycle = cycle;
                job_in = 0; job_out = 0; job_lasts = 0; job_busy = 0; job_stalls = 0;
            end
            prev_done = done;
        end
    end

    // -------------------------------------------------------------------------
    // Driver (source) and receiver (sink)
    // -------------------------------------------------------------------------
    // Send the first n_beats of slot's A-then-B stream. Before each beat,
    // s_valid may stay low for a while (gap_pct chance per cycle); once raised
    // it stays high until the beat is accepted. s_ready depends only on the
    // DUT's state, so sampling it at the falling edge tells whether the beat
    // transfers on the next rising edge.
    task automatic send(input int slot, input int gap_pct, input int n_beats);
        int value;
        bit taken;
        for (int e = 0; e < n_beats; e++) begin
            while (src_chance(gap_pct)) begin
                s_valid = 1'b0;
                @(negedge clk);
            end
            value   = (e < M*K) ? a_mat[slot][e / K][e % K]
                                : b_mat[slot][(e - M*K) / N][(e - M*K) % N];
            s_valid = 1'b1;
            s_data  = DATA_WIDTH'(value);
            do begin
                taken = s_ready;
                @(negedge clk);
            end while (!taken);
        end
        s_valid = 1'b0;
    endtask

    // Receive slot's C, checking each element and m_last as it arrives.
    task automatic receive(input int slot, input string name, input int stall_pct);
        int     idx = 0;
        int     bad = 0;
        longint got;
        bit     taken;
        while (idx < M*N) begin
            m_ready = !sink_chance(stall_pct);
            taken   = m_valid && m_ready;
            if (taken) begin
                got = longint'(m_data);
                c_got[slot][idx / N][idx % N] = got;
                n_checks += 2;
                if ($isunknown(m_data) || got != c_exp[slot][idx / N][idx % N]) begin
                    bad++;
                    if (bad <= 3)
                        error($sformatf("%s: C[%0d][%0d] = %0d, expected %0d",
                                        name, idx / N, idx % N, got, c_exp[slot][idx / N][idx % N]));
                end
                if (m_last !== (idx == M*N - 1))
                    error($sformatf("%s: m_last = %b on element %0d of %0d", name, m_last, idx, M*N));
                if (got > 0) cov_pos++;
                if (got < 0) cov_neg++;
                if (got == 0) cov_zero++;
                if (got == longint'(K) * 16384)  cov_max++;   // largest possible element
                if (got == longint'(K) * -16256) cov_min++;   // most negative possible element
                idx++;
            end
            @(negedge clk);
        end
        m_ready = 1'b0;
        if (bad > 3) error($sformatf("%s: %0d mismatching elements in total", name, bad));
    endtask

    // Append one job's A, B and received C to the results file.
    task automatic log_result(input int slot, input string name);
        if (results_fd == 0) return;
        $fwrite(results_fd, "case %s\nA", name);
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) $fwrite(results_fd, " %0d", a_mat[slot][i][k]);
        $fwrite(results_fd, "\nB");
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) $fwrite(results_fd, " %0d", b_mat[slot][k][j]);
        $fwrite(results_fd, "\nC");
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) $fwrite(results_fd, " %0d", c_got[slot][i][j]);
        $fwrite(results_fd, "\n");
    endtask

    // Called at the falling edge after a job's last output beat.
    task automatic finish_job(input int slot, input string name);
        n_checks++;
        if (done !== 1'b1) error($sformatf("%s: done not asserted after the last beat", name));
        log_result(slot, name);
        n_jobs++;
        @(negedge clk);
        n_checks++;
        if (done !== 1'b0 || busy !== 1'b0) error($sformatf("%s: done or busy still high a cycle later", name));
    endtask

    task automatic run_job(input string name, input int gap_pct, input int stall_pct);
        compute_expected(0);
        fork
            send(0, gap_pct, LOAD_BEATS);
            receive(0, name, stall_pct);
        join
        finish_job(0, name);
    endtask

    // Hand-computed expectation: every received element of C equals value.
    task automatic expect_all(input longint value, input string name);
        int bad = 0;
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) if (c_got[0][i][j] != value) bad++;
        n_checks++;
        if (bad != 0) error($sformatf("%s: %0d elements differ from the hand value %0d", name, bad, value));
        else          $display("  pass  %s: all %0d elements = %0d", name, M * N, value);
    endtask

    task automatic apply_reset(input string where);
        rst = 1'b1;
        @(negedge clk);
        rst = 1'b0;
        n_checks++;
        if (busy !== 1'b0 || s_ready !== 1'b0 || m_valid !== 1'b0 || done !== 1'b0)
            error($sformatf("after reset in %s: busy=%b s_ready=%b m_valid=%b done=%b",
                            where, busy, s_ready, m_valid, done));
    endtask

    // -------------------------------------------------------------------------
    // Reporting
    // -------------------------------------------------------------------------
    task automatic section_begin(input string name);
        sec_checks = n_checks;
        sec_errors = n_errors;
        sec_jobs   = n_jobs;
        $display("");
        $display("[%s]", name);
    endtask

    task automatic section_end();
        $display("  -> %0d jobs, %0d checks, %0d errors",
                 n_jobs - sec_jobs, n_checks - sec_checks, n_errors - sec_errors);
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
            $fwrite(results_fd, "end %0d\n", n_jobs);
            $fclose(results_fd);
            results_fd = 0;
        end
        $display("");
        if (n_errors == 0) begin
            $display("TEST PASSED%s: matrix_mult_tb M=%0d K=%0d N=%0d NUM_MACS=%0d seed=%0d | %0d jobs, %0d checks, 0 errors, %0d cycles per job back to back",
                     run_note, M, K, N, NUM_MACS, seed, n_jobs, n_checks, PERIOD);
            $finish;
        end else begin
            $display("TEST FAILED%s: matrix_mult_tb M=%0d K=%0d N=%0d NUM_MACS=%0d seed=%0d | %0d errors in %0d checks",
                     run_note, M, K, N, NUM_MACS, seed, n_errors, n_checks);
            $fatal(1, "matrix_mult_tb failed");
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sections
    // -------------------------------------------------------------------------
    // A job with no stalls: busy must be exact, and when it directly follows
    // another job (back_to_back), so must the spacing between their done pulses.
    task automatic directed_job(input string name, input bit back_to_back);
        run_job(name, 0, 0);
        n_checks++;
        if (last_busy != BUSY_CYCLES) error($sformatf("%s: busy %0d cycles, expected %0d", name, last_busy, BUSY_CYCLES));
        if (back_to_back && last_period != longint'(PERIOD))
            error($sformatf("%s: %0d cycles since the previous job, expected %0d", name, last_period, PERIOD));
    endtask

    task automatic test_directed();
        section_begin("1. directed matrices, no stalls, back to back");

        // Ones on the diagonal: I x B = B (rows beyond K are zero) and A x I = A
        fill_random(0);
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[0][i][k] = (i == k) ? 1 : 0;
        directed_job("identity x random", 1'b0);          // first job after reset
        fill_random(0);
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[0][k][j] = (k == j) ? 1 : 0;
        directed_job("random x identity", 1'b1);

        fill_const(0, 0);
        directed_job("zeros", 1'b1);
        expect_all(0, "zeros");

        fill_const(127, 127);
        directed_job("all 127", 1'b1);
        expect_all(longint'(K) * 16129, "all 127");       // K * 127 * 127

        fill_const(-128, -128);
        directed_job("all -128", 1'b1);
        expect_all(longint'(K) * 16384, "all -128");      // largest possible sum

        fill_const(-128, 127);
        directed_job("-128 x 127", 1'b1);
        expect_all(longint'(K) * -16256, "-128 x 127");   // most negative possible sum

        // Checkerboard of the extremes: mixed signs in every dot product
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[0][i][k] = ((i + k) % 2 == 0) ? 127 : -128;
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[0][k][j] = ((k + j) % 2 == 0) ? -128 : 127;
        directed_job("checkerboard of extremes", 1'b1);

        // Distinct, mixed-sign elements (up to 64 per matrix): catches swapped
        // or misrouted indices
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) a_mat[0][i][k] = (4*(i*K + k)) % 256 - 128;
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) b_mat[0][k][j] = 127 - (3*(k*N + j)) % 256;
        directed_job("index pattern", 1'b1);

        $display("  pass  every job: busy %0d cycles; back-to-back jobs %0d cycles apart", BUSY_CYCLES, PERIOD);
        section_end();
    endtask

    task automatic test_protocol();
        section_begin("2. protocol: stalls, a queued second job, resets");

        fill_random(0);
        run_job("input gaps", 50, 0);
        $display("  pass  input gaps (s_valid low 50%% of cycles): busy %0d cycles", last_busy);
        fill_random(0);
        run_job("output backpressure", 0, 50);
        $display("  pass  output backpressure (m_ready low 50%%): busy %0d cycles", last_busy);
        fill_random(0);
        run_job("gaps and backpressure", 70, 70);
        $display("  pass  both (70%%): busy %0d cycles", last_busy);

        // The source offers the second job's first beat as soon as the first
        // job is loaded; it must wait (s_ready low) until the next LOAD.
        fill_random(0);
        compute_expected(0);
        fill_random(1);
        compute_expected(1);
        fork
            begin
                send(0, 0, LOAD_BEATS);
                send(1, 0, LOAD_BEATS);
            end
            begin
                receive(0, "queued: first job", 0);
                finish_job(0, "queued: first job");
                receive(1, "queued: second job", 0);
                finish_job(1, "queued: second job");
            end
        join
        $display("  pass  second job queued during the first: accepted only after the first finished");

        // Reset during LOAD, COMPUTE and WRITEBACK; the next job must be exact
        fill_random(0);
        send(0, 0, LOAD_BEATS / 2);
        n_checks++;
        if (s_ready !== 1'b1) error("expected to be in LOAD");
        apply_reset("LOAD");
        cov_reset_load++;
        fill_random(0);
        directed_job("after reset in LOAD", 1'b0);

        fill_random(0);
        send(0, 0, LOAD_BEATS);                        // now in the first COMPUTE cycle
        repeat ((K > 1) ? K / 2 : 0) @(negedge clk);
        n_checks++;
        if (!(busy && !s_ready && !m_valid && !done)) error("expected to be in COMPUTE");
        apply_reset("COMPUTE");
        cov_reset_compute++;
        fill_random(0);
        directed_job("after reset in COMPUTE", 1'b0);

        fill_random(0);
        m_ready = 1'b0;
        send(0, 0, LOAD_BEATS);
        while (m_valid !== 1'b1) @(negedge clk);       // first element offered ...
        repeat (3) @(negedge clk);                      // ... and held
        apply_reset("WRITEBACK");
        cov_reset_writeback++;
        fill_random(0);
        directed_job("after reset in WRITEBACK", 1'b0);
        $display("  pass  reset in LOAD, COMPUTE and WRITEBACK: back to IDLE; each next job exact");

        section_end();
    endtask

    task automatic test_random();
        logic [31:0] r;
        section_begin($sformatf("3. random (%0d jobs, seed %0d)", N_RANDOM, seed));
        for (int t = 0; t < N_RANDOM; t++) begin
            fill_random(0);
            r = rand32();
            repeat (int'(r[5:4])) @(negedge clk);       // 0-3 idle cycles before some jobs
            run_job($sformatf("random %0d", t), stall_rate(r[1:0]), stall_rate(r[3:2]));
        end
        section_end();
    endtask

    task automatic report_coverage();
        section_begin("coverage (whole run)");
        cover_bin("C elements > 0",                     cov_pos);
        cover_bin("C elements < 0",                     cov_neg);
        cover_bin("C elements = 0",                     cov_zero);
        cover_bin("C element = K*16384 (largest)",      cov_max);
        cover_bin("C element = K*-16256 (most neg.)",   cov_min);
        cover_bin("input gap cycles (s_valid low)",     cov_gap);
        cover_bin("backpressure cycles (m_ready low)",  cov_backpressure);
        cover_bin("source waiting while busy",          cov_source_wait);
        cover_bin("back-to-back job starts",            cov_back_to_back);
        cover_bin("reset during LOAD",                  cov_reset_load);
        cover_bin("reset during COMPUTE",               cov_reset_compute);
        cover_bin("reset during WRITEBACK",             cov_reset_writeback);
    endtask

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    initial begin : main
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        rng_state = (seed == 0) ? 32'h1 : seed;   // xorshift32 must not start at 0
        rng_src   = xorshift32(rng_state ^ 32'h9E3779B9);
        rng_sink  = xorshift32(rng_state ^ 32'h7F4A7C15);

        if ($value$plusargs("resultsfile=%s", resultsfile)) begin
            results_fd = $fopen(resultsfile, "w");
            if (results_fd == 0) $fatal(1, "matrix_mult_tb: cannot open %s", resultsfile);
            $fwrite(results_fd, "# matrix_mult_tb: A, B and the C streamed out of the DUT, per job\n");
            $fwrite(results_fd, "dims %0d %0d %0d\n", M, K, N);
        end
        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, matrix_mult_tb);
        end

        $display("matrix_mult_tb: C[%0dx%0d] = A[%0dx%0d] x B[%0dx%0d], streamed; %0d busy cycles per job without stalls",
                 M, N, M, K, K, N, BUSY_CYCLES);

        rst     = 1'b1;
        s_valid = 1'b0;
        s_data  = '0;
        m_ready = 1'b0;
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

    // Watchdog: fail rather than hang if the DUT stops making progress.
    initial begin : watchdog
        repeat ((N_RANDOM + 32) * 20 * PERIOD) @(posedge clk);
        $display("TEST FAILED: matrix_mult_tb watchdog timeout");
        $fatal(1, "matrix_mult_tb timeout");
    end

endmodule

`default_nettype wire
