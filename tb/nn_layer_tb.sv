// =============================================================================
// nn_layer_tb.sv - testbench for the fully connected layer Y = ReLU(X*W + B)
// =============================================================================
// Streams X and W into the layer, holds the biases on the parallel port, and
// takes Y out, with stalls on either side. It checks:
//   * values: every Y element, in order, against a 64-bit reference model
//     (dot product, bias added with the documented INT32 wrap, then ReLU),
//     plus the position of m_last
//   * protocol (monitor, every rising edge): an offered output holds steady
//     while m_ready is low; each inference takes exactly INPUT_SIZE +
//     INPUT_SIZE*OUTPUT_SIZE input beats and OUTPUT_SIZE output beats; done is
//     a one-cycle pulse
//   * timing: busy lasts the documented number of cycles plus one per stall
//     cycle; back-to-back inferences without stalls are 282 cycles apart
//     (16 inputs, 8 outputs)
//
// Sections
//   1. directed   zeros, bias only, all 127, sums clamped by ReLU, a bias that
//                 rescues or sinks a sum (including the exact zero crossing),
//                 identity weights, and both sides of the INT32 wrap edge
//   2. protocol   input gaps, output backpressure, both; reset during LOAD,
//                 COMPUTE and WRITEBACK
//   3. random     N_RANDOM inferences, random X, W and bias
//   +  coverage
//
// Completed inferences are written to the results file for
// python/verify_results.py (NumPy golden model). The one deliberately
// wrapping case is not logged: the golden model refuses out-of-range sums
// rather than wrapping, so the wrap is checked here only.
//
// Plusargs:   +seed=<n>  +resultsfile=<path>
//             +dumpfile=<path>  waveform run: directed section only
//             +dumpall          with +dumpfile, run and record everything
// Parameters: INPUT_SIZE, OUTPUT_SIZE, N_RANDOM
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module nn_layer_tb;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    parameter int INPUT_SIZE  = 16;
    parameter int OUTPUT_SIZE = 8;
    parameter bit APPLY_RELU  = 1'b1;   // 0 exercises an output layer (raw logits)
    parameter int N_RANDOM    = 200;

    localparam int DATA_WIDTH  = 8;
    localparam int ACC_WIDTH   = 32;
    localparam int CLK_PERIOD  = 10;
    localparam int MAX_ERRORS  = 10;
    localparam int LOAD_BEATS  = INPUT_SIZE + INPUT_SIZE*OUTPUT_SIZE;
    localparam int BUSY_CYCLES = LOAD_BEATS + OUTPUT_SIZE*(INPUT_SIZE + 1) + 1;
    localparam int PERIOD      = BUSY_CYCLES + 1;

    localparam longint ACC_MAX   = (longint'(1) << (ACC_WIDTH - 1)) - 1;
    localparam longint ACC_MIN   = -(longint'(1) << (ACC_WIDTH - 1));
    localparam longint DOT_MAX   = longint'(INPUT_SIZE) * 16384;   // largest |X.W|
    localparam longint SAFE_BIAS = ACC_MAX - DOT_MAX;              // |B| <= this never wraps

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic                             clk = 1'b0;
    logic                             rst;
    logic                             s_valid;
    logic                             s_ready;
    logic signed [DATA_WIDTH-1:0]     s_data;
    logic [OUTPUT_SIZE*ACC_WIDTH-1:0] bias_flat;
    logic                             m_valid;
    logic                             m_ready;
    logic signed [ACC_WIDTH-1:0]      m_data;
    logic                             m_last;
    logic                             busy;
    logic                             done;

    nn_layer #(
        .INPUT_SIZE  (INPUT_SIZE),
        .OUTPUT_SIZE (OUTPUT_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .APPLY_RELU  (APPLY_RELU)
    ) dut (
        .clk       (clk),
        .rst       (rst),
        .s_valid   (s_valid),
        .s_ready   (s_ready),
        .s_data    (s_data),
        .bias_flat (bias_flat),
        .m_valid   (m_valid),
        .m_ready   (m_ready),
        .m_data    (m_data),
        .m_last    (m_last),
        .busy      (busy),
        .done      (done)
    );

    always #(CLK_PERIOD / 2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Stimulus data and bookkeeping (everything initialized)
    // -------------------------------------------------------------------------
    int     x_vec [INPUT_SIZE];
    int     w_mat [INPUT_SIZE][OUTPUT_SIZE];
    longint bias_v [OUTPUT_SIZE];
    longint y_exp [OUTPUT_SIZE];
    longint y_got [OUTPUT_SIZE];

    int unsigned n_checks = 0, n_errors = 0, n_jobs = 0, n_logged = 0;
    int unsigned sec_checks = 0, sec_errors = 0, sec_jobs = 0;
    int unsigned seed        = 1;
    logic [31:0] rng_state   = 32'h1;   // main sequence
    logic [31:0] rng_src     = 32'h1;   // source stalls (concurrent process)
    logic [31:0] rng_sink    = 32'h1;   // sink stalls (concurrent process)
    string       dumpfile    = "";
    string       resultsfile = "";
    string       run_note    = "";
    int          results_fd  = 0;

    // Monitor state
    longint      cycle = 0;
    int unsigned job_in = 0, job_out = 0, job_lasts = 0, job_busy = 0, job_stalls = 0;
    int unsigned last_busy = 0;
    longint      last_done_cycle = -1, last_period = 0;
    bit          prev_stalled = 1'b0, prev_done = 1'b0;
    logic signed [ACC_WIDTH-1:0] prev_m_data = '0;
    logic                        prev_m_last = 1'b0;

    // Coverage
    int unsigned cov_y_pos = 0, cov_y_clamped = 0, cov_y_zero_exact = 0;
    int unsigned cov_bias_pos = 0, cov_bias_neg = 0, cov_bias_zero = 0;
    int unsigned cov_dot_pos = 0, cov_dot_neg = 0, cov_wrap = 0;
    int unsigned cov_gap = 0, cov_backpressure = 0;
    int unsigned cov_reset_load = 0, cov_reset_compute = 0, cov_reset_writeback = 0;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
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

    function automatic int rand_operand();      // one draw in four is a corner value
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

    // Biases: a mix of small, sum-sized (so ReLU decides both ways) and extreme
    function automatic longint rand_bias();
        logic [31:0] r;
        longint      magnitude;
        longint      value;
        r = rand32();
        case (r[1:0])
            2'd0:    magnitude = longint'(r[20:8]);                       // small
            2'd1:    magnitude = DOT_MAX - longint'(r[18:8]);             // near the sum size
            2'd2:    magnitude = SAFE_BIAS - longint'(r[18:8]);           // near the safe edge
            default: magnitude = longint'(r[27:8]);
        endcase
        value = r[2] ? magnitude : -magnitude;
        if (value > SAFE_BIAS)  value = SAFE_BIAS;
        if (value < -SAFE_BIAS) value = -SAFE_BIAS;
        return value;
    endfunction

    function automatic int stall_rate(input logic [1:0] sel);
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

    // -------------------------------------------------------------------------
    // Stimulus fill and reference model
    // -------------------------------------------------------------------------
    task automatic fill_random();
        for (int k = 0; k < INPUT_SIZE; k++) x_vec[k] = rand_operand();
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < OUTPUT_SIZE; j++) w_mat[k][j] = rand_operand();
        for (int j = 0; j < OUTPUT_SIZE; j++) bias_v[j] = rand_bias();
    endtask

    task automatic fill_const(input int x_value, input int w_value, input longint b_value);
        for (int k = 0; k < INPUT_SIZE; k++) x_vec[k] = x_value;
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < OUTPUT_SIZE; j++) w_mat[k][j] = w_value;
        for (int j = 0; j < OUTPUT_SIZE; j++) bias_v[j] = b_value;
    endtask

    // Two's-complement wrap to ACC_WIDTH bits, as the hardware adder does
    function automatic longint wrap_acc(input longint value);
        logic signed [ACC_WIDTH-1:0] truncated;
        truncated = value[ACC_WIDTH-1:0];
        return longint'(truncated);
    endfunction

    // Y[j] = ReLU(wrap(X . W[:,j] + B[j])); also drives the bias port and coverage
    task automatic compute_expected();
        longint dot, total;
        for (int j = 0; j < OUTPUT_SIZE; j++) begin
            dot = 0;
            for (int k = 0; k < INPUT_SIZE; k++)
                dot += longint'(x_vec[k]) * longint'(w_mat[k][j]);
            total = wrap_acc(dot + bias_v[j]);
            if (total != dot + bias_v[j]) cov_wrap++;
            y_exp[j] = (APPLY_RELU && total < 0) ? 0 : total;

            if (dot > 0)        cov_dot_pos++;
            if (dot < 0)        cov_dot_neg++;
            if (bias_v[j] > 0)  cov_bias_pos++;
            if (bias_v[j] < 0)  cov_bias_neg++;
            if (bias_v[j] == 0) cov_bias_zero++;
            if (y_exp[j] > 0)   cov_y_pos++;
            if (total < 0)      cov_y_clamped++;
            if (total == 0)     cov_y_zero_exact++;

            bias_flat[j*ACC_WIDTH +: ACC_WIDTH] = ACC_WIDTH'(bias_v[j]);
        end
    endtask

    // -------------------------------------------------------------------------
    // Monitor: protocol and cycle accounting (sampled before the DUT updates)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        cycle++;
        if (rst) begin
            job_in = 0; job_out = 0; job_lasts = 0; job_busy = 0; job_stalls = 0;
            prev_stalled = 1'b0;
            prev_done    = 1'b0;
        end else begin
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

            if (done) begin
                n_checks++;
                if (prev_done) error("done high for more than one cycle");
                if (job_in != LOAD_BEATS || job_out != OUTPUT_SIZE || job_lasts != 1)
                    error($sformatf("inference had %0d input beats, %0d output beats, %0d m_last (expected %0d, %0d, 1)",
                                    job_in, job_out, job_lasts, LOAD_BEATS, OUTPUT_SIZE));
                if (job_busy != BUSY_CYCLES + job_stalls)
                    error($sformatf("inference busy for %0d cycles, expected %0d + %0d stalls",
                                    job_busy, BUSY_CYCLES, job_stalls));
                last_busy       = job_busy;
                last_period     = (last_done_cycle >= 0) ? cycle - last_done_cycle : 0;
                last_done_cycle = cycle;
                job_in = 0; job_out = 0; job_lasts = 0; job_busy = 0; job_stalls = 0;
            end
            prev_done = done;
        end
    end

    // -------------------------------------------------------------------------
    // Source and sink
    // -------------------------------------------------------------------------
    // X, then W row-major. s_ready depends only on the DUT's state, so sampling
    // it at the falling edge tells whether the beat transfers on the next edge.
    task automatic send(input int gap_pct, input int n_beats);
        int value;
        bit taken;
        for (int e = 0; e < n_beats; e++) begin
            while (src_chance(gap_pct)) begin
                s_valid = 1'b0;
                @(negedge clk);
            end
            value   = (e < INPUT_SIZE) ? x_vec[e]
                                       : w_mat[(e - INPUT_SIZE) / OUTPUT_SIZE][(e - INPUT_SIZE) % OUTPUT_SIZE];
            s_valid = 1'b1;
            s_data  = DATA_WIDTH'(value);
            do begin
                taken = s_ready;
                @(negedge clk);
            end while (!taken);
        end
        s_valid = 1'b0;
    endtask

    task automatic receive(input string name, input int stall_pct);
        int     idx = 0;
        int     bad = 0;
        longint got;
        bit     taken;
        while (idx < OUTPUT_SIZE) begin
            m_ready = !sink_chance(stall_pct);
            taken   = m_valid && m_ready;
            if (taken) begin
                got        = longint'(m_data);
                y_got[idx] = got;
                n_checks += 2;
                if ($isunknown(m_data) || got != y_exp[idx]) begin
                    bad++;
                    if (bad <= 3)
                        error($sformatf("%s: Y[%0d] = %0d, expected %0d", name, idx, got, y_exp[idx]));
                end
                if (m_last !== (idx == OUTPUT_SIZE - 1))
                    error($sformatf("%s: m_last = %b on output %0d of %0d", name, m_last, idx, OUTPUT_SIZE));
                idx++;
            end
            @(negedge clk);
        end
        m_ready = 1'b0;
        if (bad > 3) error($sformatf("%s: %0d mismatching outputs in total", name, bad));
    endtask

    task automatic log_result(input string name);
        if (results_fd == 0) return;
        n_logged++;
        $fwrite(results_fd, "case %s\nX", name);
        for (int k = 0; k < INPUT_SIZE; k++) $fwrite(results_fd, " %0d", x_vec[k]);
        $fwrite(results_fd, "\nW");
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < OUTPUT_SIZE; j++) $fwrite(results_fd, " %0d", w_mat[k][j]);
        $fwrite(results_fd, "\nB");
        for (int j = 0; j < OUTPUT_SIZE; j++) $fwrite(results_fd, " %0d", bias_v[j]);
        $fwrite(results_fd, "\nY");
        for (int j = 0; j < OUTPUT_SIZE; j++) $fwrite(results_fd, " %0d", y_got[j]);
        $fwrite(results_fd, "\n");
    endtask

    // log = 0 for the deliberately wrapping case, which the NumPy model rejects
    task automatic run_job(input string name, input int gap_pct, input int stall_pct, input bit log);
        compute_expected();
        fork
            send(gap_pct, LOAD_BEATS);
            receive(name, stall_pct);
        join
        n_checks++;
        if (done !== 1'b1) error($sformatf("%s: done not asserted after the last output", name));
        if (log) log_result(name);
        n_jobs++;
        @(negedge clk);
        n_checks++;
        if (done !== 1'b0 || busy !== 1'b0) error($sformatf("%s: done or busy still high a cycle later", name));
    endtask

    // An inference with no stalls: busy and (optionally) the spacing must be exact
    task automatic directed_job(input string name, input bit back_to_back);
        run_job(name, 0, 0, 1'b1);
        n_checks++;
        if (last_busy != BUSY_CYCLES)
            error($sformatf("%s: busy %0d cycles, expected %0d", name, last_busy, BUSY_CYCLES));
        if (back_to_back && last_period != longint'(PERIOD))
            error($sformatf("%s: %0d cycles since the previous inference, expected %0d",
                            name, last_period, PERIOD));
    endtask

    task automatic expect_outputs(input longint value, input string name);
        int bad = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) if (y_got[j] != value) bad++;
        n_checks++;
        if (bad != 0) error($sformatf("%s: %0d outputs differ from the hand value %0d", name, bad, value));
        else          $display("  pass  %s: all %0d outputs = %0d", name, OUTPUT_SIZE, value);
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
        $display("  -> %0d inferences, %0d checks, %0d errors",
                 n_jobs - sec_jobs, n_checks - sec_checks, n_errors - sec_errors);
    endtask

    task automatic cover_bin(input string name, input int unsigned hits, input bit required);
        if (!required) begin
            $display("       n/a  %s (not applicable in this configuration)", name);
        end else begin
            $display("  %8d  %s", hits, name);
            if (hits == 0) begin
                n_errors++;
                $display("ERROR coverage hole: '%s' was never exercised", name);
            end
        end
    endtask

    task automatic end_test();
        if (results_fd != 0) begin
            $fwrite(results_fd, "end %0d\n", n_logged);   // logged cases, not inferences
            $fclose(results_fd);
            results_fd = 0;
        end
        $display("");
        if (n_errors == 0) begin
            $display("TEST PASSED%s: nn_layer_tb INPUT_SIZE=%0d OUTPUT_SIZE=%0d APPLY_RELU=%0d seed=%0d | %0d inferences, %0d checks, 0 errors, %0d cycles per inference back to back",
                     run_note, INPUT_SIZE, OUTPUT_SIZE, APPLY_RELU, seed, n_jobs, n_checks, PERIOD);
            $finish;
        end else begin
            $display("TEST FAILED%s: nn_layer_tb INPUT_SIZE=%0d OUTPUT_SIZE=%0d APPLY_RELU=%0d seed=%0d | %0d errors in %0d checks",
                     run_note, INPUT_SIZE, OUTPUT_SIZE, APPLY_RELU, seed, n_errors, n_checks);
            $fatal(1, "nn_layer_tb failed");
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sections
    // -------------------------------------------------------------------------
    task automatic test_directed();
        longint dot_all_127, dot_min_max;
        section_begin("1. directed, no stalls, back to back");

        dot_all_127 = longint'(INPUT_SIZE) * 127 * 127;    // 258,064 for 16 inputs
        dot_min_max = longint'(INPUT_SIZE) * -128 * 127;   // -260,096

        fill_const(0, 0, 0);
        directed_job("zeros", 1'b0);                        // first inference after reset
        expect_outputs(0, "zeros");

        // Bias only: X = 0, so Y = ReLU(B); alternate sign so ReLU clamps half
        for (int k = 0; k < INPUT_SIZE; k++) x_vec[k] = 0;
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < OUTPUT_SIZE; j++) w_mat[k][j] = rand_operand();
        for (int j = 0; j < OUTPUT_SIZE; j++)
            bias_v[j] = (j % 2 == 0) ? (1000 + longint'(j)) : -(1000 + longint'(j));
        directed_job("bias only", 1'b1);

        fill_const(127, 127, 0);
        directed_job("all 127", 1'b1);
        expect_outputs(dot_all_127, "all 127");

        fill_const(-128, 127, 0);
        directed_job(APPLY_RELU ? "negative sums clamp to 0" : "negative sums pass through", 1'b1);
        expect_outputs(APPLY_RELU ? 0 : dot_min_max,
                       APPLY_RELU ? "negative sums clamp to 0" : "negative sums pass through");

        // A bias that lifts a negative sum back up: Y[j] = j, so Y[0] is the
        // exact zero crossing
        fill_const(-128, 127, 0);
        for (int j = 0; j < OUTPUT_SIZE; j++) bias_v[j] = -dot_min_max + longint'(j);
        directed_job("bias rescues a negative sum", 1'b1);
        for (int j = 0; j < OUTPUT_SIZE; j++) begin
            n_checks++;
            if (y_got[j] != longint'(j)) error($sformatf("bias rescue: Y[%0d] = %0d, expected %0d", j, y_got[j], j));
        end
        $display("  pass  bias rescues a negative sum: Y[j] = j, including Y[0] = 0 at the crossing");

        // A bias that sinks a positive sum: Y[j] = j again, from the other side
        fill_const(127, 127, 0);
        for (int j = 0; j < OUTPUT_SIZE; j++) bias_v[j] = -dot_all_127 + longint'(j);
        directed_job("bias sinks a positive sum", 1'b1);
        for (int j = 0; j < OUTPUT_SIZE; j++) begin
            n_checks++;
            if (y_got[j] != longint'(j)) error($sformatf("bias sink: Y[%0d] = %0d, expected %0d", j, y_got[j], j));
        end

        // Identity weights: Y[j] = ReLU(X[j] + B[j]) for j < INPUT_SIZE
        for (int k = 0; k < INPUT_SIZE; k++) x_vec[k] = rand_operand();
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < OUTPUT_SIZE; j++) w_mat[k][j] = (k == j) ? 1 : 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) bias_v[j] = rand_bias();
        directed_job("identity weights", 1'b1);

        // The documented INT32 edge: largest sum plus the largest safe bias
        fill_const(-128, -128, SAFE_BIAS);                  // dot = +DOT_MAX
        directed_job("largest sum + largest safe bias", 1'b1);
        expect_outputs(ACC_MAX, "largest sum + largest safe bias");

        // One more wraps to the most negative value, and ReLU clamps it to 0
        fill_const(-128, -128, SAFE_BIAS + 1);
        run_job("one past the safe bias wraps (not logged)", 0, 0, 1'b0);
        expect_outputs(APPLY_RELU ? 0 : ACC_MIN,
                       APPLY_RELU ? "one past the safe bias wraps to negative, clamped to 0"
                                  : "one past the safe bias wraps to the most negative value");

        $display("  pass  every inference: busy %0d cycles; back to back %0d cycles apart",
                 BUSY_CYCLES, PERIOD);
        section_end();
    endtask

    task automatic test_protocol();
        section_begin("2. protocol: stalls and resets");

        fill_random();
        run_job("input gaps", 50, 0, 1'b1);
        $display("  pass  input gaps (s_valid low 50%%): busy %0d cycles", last_busy);
        fill_random();
        run_job("output backpressure", 0, 50, 1'b1);
        $display("  pass  output backpressure (m_ready low 50%%): busy %0d cycles", last_busy);
        fill_random();
        run_job("gaps and backpressure", 70, 70, 1'b1);
        $display("  pass  both (70%%): busy %0d cycles", last_busy);

        fill_random();
        compute_expected();
        send(0, LOAD_BEATS / 2);
        n_checks++;
        if (s_ready !== 1'b1) error("expected to be in LOAD");
        apply_reset("LOAD");
        cov_reset_load++;
        fill_random();
        directed_job("after reset in LOAD", 1'b0);

        fill_random();
        compute_expected();
        send(0, LOAD_BEATS);                                  // now computing
        repeat (INPUT_SIZE / 2) @(negedge clk);
        n_checks++;
        if (!(busy && !s_ready && !m_valid && !done)) error("expected to be computing");
        apply_reset("COMPUTE");
        cov_reset_compute++;
        fill_random();
        directed_job("after reset in COMPUTE", 1'b0);

        fill_random();
        compute_expected();
        m_ready = 1'b0;
        send(0, LOAD_BEATS);
        while (m_valid !== 1'b1) @(negedge clk);              // first output offered ...
        repeat (3) @(negedge clk);                            // ... and held
        apply_reset("WRITEBACK");
        cov_reset_writeback++;
        fill_random();
        directed_job("after reset in WRITEBACK", 1'b0);
        $display("  pass  reset in LOAD, COMPUTE and WRITEBACK: back to IDLE; each next inference exact");

        section_end();
    endtask

    task automatic test_random();
        logic [31:0] r;
        section_begin($sformatf("3. random (%0d inferences, seed %0d)", N_RANDOM, seed));
        for (int t = 0; t < N_RANDOM; t++) begin
            fill_random();
            r = rand32();
            repeat (int'(r[5:4])) @(negedge clk);
            run_job($sformatf("random %0d", t), stall_rate(r[1:0]), stall_rate(r[3:2]), 1'b1);
        end
        section_end();
    endtask

    task automatic report_coverage();
        section_begin("coverage (whole run)");
        cover_bin("Y > 0",                             cov_y_pos,        1'b1);
        cover_bin("Y clamped to 0 by ReLU",            cov_y_clamped,    APPLY_RELU);
        cover_bin("X.W + B exactly 0",                 cov_y_zero_exact, 1'b1);
        cover_bin("dot product > 0",                   cov_dot_pos, 1'b1);
        cover_bin("dot product < 0",                   cov_dot_neg, 1'b1);
        cover_bin("bias > 0",                          cov_bias_pos, 1'b1);
        cover_bin("bias < 0",                          cov_bias_neg, 1'b1);
        cover_bin("bias = 0",                          cov_bias_zero, 1'b1);
        cover_bin("INT32 wrap (documented edge)",      cov_wrap, 1'b1);
        cover_bin("input gap cycles",                  cov_gap, 1'b1);
        cover_bin("backpressure cycles",               cov_backpressure, 1'b1);
        cover_bin("reset during LOAD",                 cov_reset_load, 1'b1);
        cover_bin("reset during COMPUTE",              cov_reset_compute, 1'b1);
        cover_bin("reset during WRITEBACK",            cov_reset_writeback, 1'b1);
    endtask

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    initial begin : main
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        rng_state = (seed == 0) ? 32'h1 : seed;
        rng_src   = xorshift32(rng_state ^ 32'h9E3779B9);
        rng_sink  = xorshift32(rng_state ^ 32'h7F4A7C15);

        if ($value$plusargs("resultsfile=%s", resultsfile)) begin
            results_fd = $fopen(resultsfile, "w");
            if (results_fd == 0) $fatal(1, "nn_layer_tb: cannot open %s", resultsfile);
            $fwrite(results_fd, "# nn_layer_tb: X, W, B and the Y streamed out of the DUT, per inference\n");
            $fwrite(results_fd, "dims %0d %0d %0d\n", INPUT_SIZE, OUTPUT_SIZE, APPLY_RELU);
        end
        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, nn_layer_tb);
        end

        $display("nn_layer_tb: Y = %sX*W + B%s, %0d inputs -> %0d outputs, %0d cycles per inference without stalls",
                 APPLY_RELU ? "ReLU(" : "", APPLY_RELU ? ")" : "", INPUT_SIZE, OUTPUT_SIZE, BUSY_CYCLES);

        rst       = 1'b1;
        s_valid   = 1'b0;
        s_data    = '0;
        m_ready   = 1'b0;
        bias_flat = '0;
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

    initial begin : watchdog
        repeat ((N_RANDOM + 32) * 20 * PERIOD) @(posedge clk);
        $display("TEST FAILED: nn_layer_tb watchdog timeout");
        $fatal(1, "nn_layer_tb timeout");
    end

endmodule

`default_nettype wire
