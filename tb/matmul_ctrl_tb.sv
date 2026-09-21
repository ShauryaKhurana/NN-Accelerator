// =============================================================================
// matmul_ctrl_tb.sv - unit testbench for the matrix-multiply control FSM
// =============================================================================
// The controller is tested on its own, with no datapath. Every cycle:
//   * s_valid and m_ready are driven (directed or random),
//   * every output is checked against the state of a reference model written
//     from the transition table in matmul_ctrl.sv,
//   * after the clock edge, the DUT's state and counters (ld_idx, i, j, k)
//     must match the model.
// For each job, busy must last exactly M*K + K*N + M*N*(K+1) + 1 cycles plus
// one per stall cycle (s_valid low in LOAD, m_ready low in WRITEBACK).
//
// Sections
//   1. one job with no stalls        exact cycle count
//   2. stalls                        input gaps, output backpressure, both
//   3. reset from each busy state    LOAD, COMPUTE, WRITEBACK, DONE
//   4. random jobs                   N_JOBS jobs, random stall rates and gaps
//   +  coverage                      every legal transition, every reset
// Note: s_valid and m_ready are driven as unconstrained random bits here. The
// controller only counts handshakes; the full stream protocol is exercised by
// matrix_mult_tb.sv.
//
// Plusargs:   +seed=<n>  +dumpfile=<path> (waveform run: sections 1-2 only)
// Parameters: M, K, N (dimensions), N_JOBS
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module matmul_ctrl_tb;

    import matmul_pkg::*;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    parameter int M      = 8;
    parameter int K      = 8;
    parameter int N      = 8;
    parameter int N_JOBS = 40;

    localparam int LOAD_BEATS  = M*K + K*N;
    localparam int BUSY_CYCLES = LOAD_BEATS + M*N*(K + 1) + 1;   // per job, no stalls
    localparam int LW = $clog2(LOAD_BEATS);
    localparam int IW = (M > 1) ? $clog2(M) : 1;
    localparam int JW = (N > 1) ? $clog2(N) : 1;
    localparam int KW = (K > 1) ? $clog2(K) : 1;
    localparam int MAX_ERRORS = 10;

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic          clk = 1'b0;
    logic          rst;
    logic          s_valid;
    logic          s_ready;
    logic          m_valid;
    logic          m_ready;
    logic          m_last;
    logic          busy;
    logic          done;
    logic          ld_en;
    logic [LW-1:0] ld_idx;
    logic          mac_en;
    logic          mac_clear;
    logic [IW-1:0] i;
    logic [JW-1:0] j;
    logic [KW-1:0] k;

    matmul_ctrl #(
        .M (M),
        .K (K),
        .N (N)
    ) dut (
        .clk       (clk),
        .rst       (rst),
        .s_valid   (s_valid),
        .s_ready   (s_ready),
        .m_valid   (m_valid),
        .m_ready   (m_ready),
        .m_last    (m_last),
        .busy      (busy),
        .done      (done),
        .ld_en     (ld_en),
        .ld_idx    (ld_idx),
        .mac_en    (mac_en),
        .mac_clear (mac_clear),
        .i         (i),
        .j         (j),
        .k         (k)
    );

    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Reference model (the documented transition table) and bookkeeping
    // -------------------------------------------------------------------------
    matmul_state_t exp_state = S_IDLE;
    int            exp_ld = 0, exp_i = 0, exp_j = 0, exp_k = 0;

    int unsigned n_checks = 0, n_errors = 0, n_jobs = 0, n_cycles = 0;
    int unsigned sec_checks = 0, sec_errors = 0, sec_jobs = 0;
    int unsigned job_busy = 0, job_stall = 0, last_busy = 0;
    int unsigned seed      = 1;
    logic [31:0] rng_state = 32'h1;
    string       dumpfile  = "";
    string       run_note  = "";

    int unsigned trans [5][5];         // transition counts [from][to]
    int unsigned reset_from [5];       // resets applied in each state

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    function automatic logic [31:0] rand32();
        rng_state = rng_state ^ (rng_state << 13);
        rng_state = rng_state ^ (rng_state >> 17);
        rng_state = rng_state ^ (rng_state << 5);
        return rng_state;
    endfunction

    function automatic bit chance(input int percent);
        return (rand32() % 100) < percent;
    endfunction

    function automatic string sname(input matmul_state_t s);
        case (s)
            S_IDLE:      return "IDLE";
            S_LOAD:      return "LOAD";
            S_COMPUTE:   return "COMPUTE";
            S_WRITEBACK: return "WRITEBACK";
            S_DONE:      return "DONE";
            default:     return "INVALID";
        endcase
    endfunction

    task automatic error(input string message);
        n_errors++;
        $display("ERROR cycle %0d (model state %s): %s", n_cycles, sname(exp_state), message);
        if (n_errors >= MAX_ERRORS) end_test();
    endtask

    task automatic expect_bit(input string name, input logic got, input bit want);
        n_checks++;
        if (got !== want) error($sformatf("%s = %b, expected %b", name, got, want));
    endtask

    // -------------------------------------------------------------------------
    // Model
    // -------------------------------------------------------------------------
    // Outputs during the current cycle, from the model's state
    task automatic check_outputs(input bit sv);
        expect_bit("s_ready",   s_ready,   exp_state == S_LOAD);
        expect_bit("ld_en",     ld_en,     exp_state == S_LOAD && sv);
        expect_bit("mac_en",    mac_en,    exp_state == S_COMPUTE);
        expect_bit("mac_clear", mac_clear, exp_state == S_COMPUTE && exp_k == 0);
        expect_bit("m_valid",   m_valid,   exp_state == S_WRITEBACK);
        expect_bit("m_last",    m_last,    exp_state == S_WRITEBACK && exp_i == M-1 && exp_j == N-1);
        expect_bit("busy",      busy,      exp_state != S_IDLE);
        expect_bit("done",      done,      exp_state == S_DONE);
    endtask

    // Per-job cycle accounting; a job completes in its DONE cycle
    task automatic account(input bit sv, input bit mr);
        if (exp_state != S_IDLE) job_busy++;
        if ((exp_state == S_LOAD && !sv) || (exp_state == S_WRITEBACK && !mr)) job_stall++;
        if (exp_state == S_DONE) begin
            n_checks++;
            if (job_busy != BUSY_CYCLES + job_stall)
                error($sformatf("job busy for %0d cycles, expected %0d + %0d stalls",
                                job_busy, BUSY_CYCLES, job_stall));
            last_busy = job_busy;
            job_busy  = 0;
            job_stall = 0;
            n_jobs++;
        end
    endtask

    // The transition table from matmul_ctrl.sv: state and counters after the edge
    task automatic model_advance(input bit sv, input bit mr);
        case (exp_state)
            S_IDLE: begin
                exp_ld = 0; exp_i = 0; exp_j = 0; exp_k = 0;
                if (sv) exp_state = S_LOAD;
            end
            S_LOAD:
                if (sv) begin
                    if (exp_ld == LOAD_BEATS - 1) exp_state = S_COMPUTE;
                    exp_ld++;
                end
            S_COMPUTE:
                if (exp_k == K - 1) begin
                    exp_k     = 0;
                    exp_state = S_WRITEBACK;
                end else begin
                    exp_k++;
                end
            S_WRITEBACK:
                if (mr) begin
                    if (exp_i == M - 1 && exp_j == N - 1) begin
                        exp_state = S_DONE;
                    end else begin
                        exp_state = S_COMPUTE;
                        if (exp_j == N - 1) begin
                            exp_j = 0;
                            exp_i++;
                        end else begin
                            exp_j++;
                        end
                    end
                end
            default: exp_state = S_IDLE;   // S_DONE
        endcase
    endtask

    task automatic compare_model();
        n_checks++;
        if (dut.state !== exp_state) begin
            error($sformatf("state %s, expected %s", sname(dut.state), sname(exp_state)));
        end else begin
            case (exp_state)
                S_LOAD:
                    if (int'(ld_idx) != exp_ld)
                        error($sformatf("ld_idx %0d, expected %0d", ld_idx, exp_ld));
                S_COMPUTE, S_WRITEBACK:
                    if (int'(i) != exp_i || int'(j) != exp_j || int'(k) != exp_k)
                        error($sformatf("(i,j,k) = (%0d,%0d,%0d), expected (%0d,%0d,%0d)",
                                        i, j, k, exp_i, exp_j, exp_k));
                default: ;
            endcase
        end
    endtask

    // -------------------------------------------------------------------------
    // One clock cycle: drive the inputs at a falling edge, check this cycle's
    // outputs, advance the model across the rising edge, then compare.
    // -------------------------------------------------------------------------
    task automatic step(input bit sv, input bit mr);
        matmul_state_t prev_state;
        s_valid = sv;
        m_ready = mr;
        #1;                                   // let the combinational outputs settle
        prev_state = exp_state;
        check_outputs(sv);
        account(sv, mr);
        model_advance(sv, mr);
        @(negedge clk);
        n_cycles++;
        trans[int'(prev_state)][int'(exp_state)]++;
        compare_model();
    endtask

    task automatic reset_step();
        reset_from[int'(exp_state)]++;
        rst = 1'b1;
        @(negedge clk);
        rst = 1'b0;
        n_cycles++;
        exp_state = S_IDLE;
        exp_ld = 0; exp_i = 0; exp_j = 0; exp_k = 0;
        job_busy  = 0;
        job_stall = 0;
        compare_model();
    endtask

    // Run until one more job completes; each cycle, s_valid is low with
    // probability stall_in and m_ready is low with probability stall_out (%).
    task automatic run_job(input int stall_in, input int stall_out);
        int unsigned target = n_jobs + 1;
        int          guard  = 0;
        while (n_jobs < target && guard < 50 * BUSY_CYCLES) begin
            step(!chance(stall_in), !chance(stall_out));
            guard++;
        end
        if (n_jobs < target) error("job did not complete");
    endtask

    // Start a job, run it (no stalls) until the model reaches `target`, reset
    // there, and check that a clean job afterwards is exact.
    task automatic reset_in(input matmul_state_t target);
        int guard = 0;
        while (exp_state != target && guard < 4 * BUSY_CYCLES) begin
            step(1'b1, 1'b1);
            guard++;
        end
        reset_step();
        run_job(0, 0);
        n_checks++;
        if (last_busy != BUSY_CYCLES) error("job after reset was not exact");
        $display("  pass  reset in %s -> IDLE; next job exact (%0d busy cycles)", sname(target), last_busy);
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

    task automatic cover_bin(input string name, input int unsigned hits, input bit required);
        if (required) begin
            $display("  %8d  %s", hits, name);
            if (hits == 0) begin
                n_errors++;
                $display("ERROR coverage hole: '%s' was never exercised", name);
            end
        end else begin
            $display("       n/a  %s (impossible for these dimensions)", name);
        end
    endtask

    task automatic end_test();
        $display("");
        if (n_errors == 0) begin
            $display("TEST PASSED%s: matmul_ctrl_tb M=%0d K=%0d N=%0d seed=%0d | %0d jobs, %0d cycles, %0d checks, 0 errors, %0d busy cycles per job",
                     run_note, M, K, N, seed, n_jobs, n_cycles, n_checks, BUSY_CYCLES);
            $finish;
        end else begin
            $display("TEST FAILED%s: matmul_ctrl_tb M=%0d K=%0d N=%0d seed=%0d | %0d errors in %0d checks",
                     run_note, M, K, N, seed, n_errors, n_checks);
            $fatal(1, "matmul_ctrl_tb failed");
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sections
    // -------------------------------------------------------------------------
    task automatic test_no_stalls();
        section_begin("1. one job, no stalls");
        run_job(0, 0);
        n_checks++;
        if (last_busy != BUSY_CYCLES) error("busy cycles differ from the documented count");
        $display("  pass  busy for %0d cycles = (M*K + K*N) + M*N*(K+1) + 1 = %0d + %0d + 1",
                 last_busy, LOAD_BEATS, M*N*(K + 1));
        section_end();
    endtask

    task automatic test_stalls();
        section_begin("2. stalls: every stall cycle adds exactly one cycle");
        run_job(50, 0);
        $display("  pass  input gaps (s_valid low 50%%): busy %0d cycles", last_busy);
        run_job(0, 50);
        $display("  pass  output backpressure (m_ready low 50%%): busy %0d cycles", last_busy);
        run_job(70, 70);
        $display("  pass  both (70%%): busy %0d cycles", last_busy);
        section_end();
    endtask

    task automatic test_resets();
        section_begin("3. reset from each busy state");
        reset_in(S_LOAD);
        reset_in(S_COMPUTE);
        reset_in(S_WRITEBACK);
        reset_in(S_DONE);
        section_end();
    endtask

    function automatic int stall_rate(input logic [1:0] sel);   // percent
        case (sel)
            2'd0:    return 0;
            2'd1:    return 10;
            2'd2:    return 50;
            default: return 90;
        endcase
    endfunction

    task automatic test_random();
        logic [31:0] r;
        section_begin($sformatf("4. random jobs (%0d, seed %0d)", N_JOBS, seed));
        for (int n = 0; n < N_JOBS; n++) begin
            r = rand32();
            repeat (int'(r[5:4])) step(1'b0, r[6]);        // 0-3 idle cycles between jobs
            run_job(stall_rate(r[1:0]), stall_rate(r[3:2]));
        end
        section_end();
    endtask

    task automatic report_coverage();
        section_begin("coverage: transitions and resets (cycles)");
        cover_bin("IDLE -> IDLE",           trans[S_IDLE][S_IDLE],           1'b1);
        cover_bin("IDLE -> LOAD",           trans[S_IDLE][S_LOAD],           1'b1);
        cover_bin("LOAD -> LOAD",           trans[S_LOAD][S_LOAD],           1'b1);
        cover_bin("LOAD -> COMPUTE",        trans[S_LOAD][S_COMPUTE],        1'b1);
        cover_bin("COMPUTE -> COMPUTE",     trans[S_COMPUTE][S_COMPUTE],     K > 1);
        cover_bin("COMPUTE -> WRITEBACK",   trans[S_COMPUTE][S_WRITEBACK],   1'b1);
        cover_bin("WRITEBACK -> WRITEBACK", trans[S_WRITEBACK][S_WRITEBACK], 1'b1);
        cover_bin("WRITEBACK -> COMPUTE",   trans[S_WRITEBACK][S_COMPUTE],   M*N > 1);
        cover_bin("WRITEBACK -> DONE",      trans[S_WRITEBACK][S_DONE],      1'b1);
        cover_bin("DONE -> IDLE",           trans[S_DONE][S_IDLE],           1'b1);
        cover_bin("reset in LOAD",          reset_from[S_LOAD],              1'b1);
        cover_bin("reset in COMPUTE",       reset_from[S_COMPUTE],           1'b1);
        cover_bin("reset in WRITEBACK",     reset_from[S_WRITEBACK],         1'b1);
        cover_bin("reset in DONE",          reset_from[S_DONE],              1'b1);
    endtask

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    initial begin : main
        for (int a = 0; a < 5; a++) begin
            reset_from[a] = 0;
            for (int b = 0; b < 5; b++) trans[a][b] = 0;
        end
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        rng_state = (seed == 0) ? 32'h1 : seed;
        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, matmul_ctrl_tb);
        end

        $display("matmul_ctrl_tb: M=%0d K=%0d N=%0d, %0d load beats, %0d busy cycles per job without stalls",
                 M, K, N, LOAD_BEATS, BUSY_CYCLES);

        rst     = 1'b1;
        s_valid = 1'b0;
        m_ready = 1'b0;
        repeat (2) @(negedge clk);
        rst = 1'b0;
        compare_model();

        test_no_stalls();
        test_stalls();
        if (dumpfile != "" && !$test$plusargs("dumpall")) begin
            run_note = " (waveform run: sections 1-2 only)";
            $display("");
            $display("waveform run: sections 3-4 skipped; add +dumpall to run and record them");
        end else begin
            test_resets();
            test_random();
            report_coverage();
        end
        end_test();
    end

    initial begin : watchdog
        repeat ((N_JOBS + 20) * 50 * BUSY_CYCLES) @(posedge clk);
        $display("TEST FAILED: matmul_ctrl_tb watchdog timeout");
        $fatal(1, "matmul_ctrl_tb timeout");
    end

endmodule

`default_nettype wire
