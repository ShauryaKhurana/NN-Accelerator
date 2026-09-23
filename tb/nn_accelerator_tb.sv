// =============================================================================
// nn_accelerator_tb.sv - testbench for the two-layer INT8 network
// =============================================================================
// Streams X, W1 and W2 into the accelerator, holds both bias vectors on their
// parallel ports, and takes the logits out, with stalls on either side.
// It checks:
//   * values: every logit against a 64-bit reference that repeats the whole
//     pipeline (dot product, bias, ReLU, requantize, dot product, bias), plus
//     the position of m_last
//   * protocol: an offered logit holds steady while m_ready is low; each
//     inference takes exactly INPUT_SIZE + INPUT_SIZE*HIDDEN_SIZE +
//     HIDDEN_SIZE*OUTPUT_SIZE input beats and OUTPUT_SIZE output beats; done
//     is a one-cycle pulse
//   * timing: busy lasts the documented number of cycles plus one per stall
//     cycle, and back-to-back inferences keep that spacing
//
// Sections
//   1. directed   zeros; bias-driven hidden values that walk the requantizer
//                 through 0, 1, 127 and saturation, read back directly through
//                 identity second-layer weights; all 127; hidden values that
//                 ReLU zeroes
//   2. protocol   input gaps, output backpressure, both; reset while loading
//                 layer 1, while layer 1 computes, while loading W2, and while
//                 layer 2 computes
//   3. random     N_RANDOM inferences with random X, W1, W2 and biases
//   +  coverage
//
// Completed inferences go to the results file for python/verify_results.py,
// which recomputes them with the NumPy golden model.
//
// Plusargs:   +seed=<n>  +resultsfile=<path>
//             +dumpfile=<path>  waveform run: directed section only
//             +dumpall          with +dumpfile, run and record everything
// Parameters: INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SHIFT, N_RANDOM
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module nn_accelerator_tb;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    parameter int INPUT_SIZE  = 16;
    parameter int HIDDEN_SIZE = 32;
    parameter int OUTPUT_SIZE = 10;
    parameter int SHIFT       = 8;
    parameter int NUM_MACS    = 1;
    parameter int N_RANDOM    = 100;

    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int CLK_PERIOD = 10;
    localparam int MAX_ERRORS = 10;

    localparam int L1_BEATS   = INPUT_SIZE + INPUT_SIZE*HIDDEN_SIZE;
    localparam int W2_BEATS   = HIDDEN_SIZE*OUTPUT_SIZE;
    localparam int HOST_BEATS = L1_BEATS + W2_BEATS;
    localparam int GROUPS1 = (HIDDEN_SIZE + NUM_MACS - 1) / NUM_MACS;
    localparam int GROUPS2 = (OUTPUT_SIZE + NUM_MACS - 1) / NUM_MACS;
    // Layer 1 loads and computes; the first activation waits one cycle for
    // layer 2 to leave IDLE; then W2 loads, layer 2 computes, and DONE.
    localparam int BUSY_CYCLES = L1_BEATS + GROUPS1*INPUT_SIZE + HIDDEN_SIZE + 1
                                 + W2_BEATS + GROUPS2*HIDDEN_SIZE + OUTPUT_SIZE + 1;
    localparam int PERIOD      = BUSY_CYCLES + 1;

    localparam longint ACC_MAX    = (longint'(1) << (ACC_WIDTH - 1)) - 1;
    localparam longint DOT1_MAX   = longint'(INPUT_SIZE) * 16384;
    localparam longint DOT2_MAX   = longint'(HIDDEN_SIZE) * 16384;
    localparam longint SAFE_BIAS1 = ACC_MAX - DOT1_MAX;
    localparam longint SAFE_BIAS2 = ACC_MAX - DOT2_MAX;
    localparam longint ACT_MAX    = 127;

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic                             clk = 1'b0;
    logic                             rst;
    logic                             s_valid;
    logic                             s_ready;
    logic signed [DATA_WIDTH-1:0]     s_data;
    logic [HIDDEN_SIZE*ACC_WIDTH-1:0] bias1_flat;
    logic [OUTPUT_SIZE*ACC_WIDTH-1:0] bias2_flat;
    logic                             m_valid;
    logic                             m_ready;
    logic signed [ACC_WIDTH-1:0]      m_data;
    logic                             m_last;
    logic                             busy;
    logic                             done;

    nn_accelerator #(
        .INPUT_SIZE  (INPUT_SIZE),
        .HIDDEN_SIZE (HIDDEN_SIZE),
        .OUTPUT_SIZE (OUTPUT_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .SHIFT       (SHIFT),
        .NUM_MACS    (NUM_MACS)
    ) dut (
        .clk        (clk),
        .rst        (rst),
        .s_valid    (s_valid),
        .s_ready    (s_ready),
        .s_data     (s_data),
        .bias1_flat (bias1_flat),
        .bias2_flat (bias2_flat),
        .m_valid    (m_valid),
        .m_ready    (m_ready),
        .m_data     (m_data),
        .m_last     (m_last),
        .busy       (busy),
        .done       (done)
    );

    always #(CLK_PERIOD / 2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Stimulus and reference state
    // -------------------------------------------------------------------------
    int     x_vec  [INPUT_SIZE];
    int     w1_mat [INPUT_SIZE][HIDDEN_SIZE];
    longint b1_vec [HIDDEN_SIZE];
    int     w2_mat [HIDDEN_SIZE][OUTPUT_SIZE];
    longint b2_vec [OUTPUT_SIZE];
    longint act_exp [HIDDEN_SIZE];      // requantized hidden activations
    longint y_exp  [OUTPUT_SIZE];
    longint y_got  [OUTPUT_SIZE];

    int unsigned n_checks = 0, n_errors = 0, n_jobs = 0, n_logged = 0;
    int unsigned sec_checks = 0, sec_errors = 0, sec_jobs = 0;
    int unsigned seed        = 1;
    logic [31:0] rng_state   = 32'h1;
    logic [31:0] rng_src     = 32'h1;
    logic [31:0] rng_sink    = 32'h1;
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
    int unsigned cov_logit_pos = 0, cov_logit_neg = 0, cov_logit_zero = 0;
    int unsigned cov_act_sat = 0, cov_act_zero = 0, cov_act_mid = 0, cov_relu_zeroed = 0;
    int unsigned cov_gap = 0, cov_backpressure = 0;
    int unsigned cov_reset_l1_load = 0, cov_reset_l1_calc = 0;
    int unsigned cov_reset_w2_load = 0, cov_reset_l2_calc = 0;

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

    function automatic int stall_rate(input logic [1:0] sel);
        case (sel)
            2'd0:    return 0;
            2'd1:    return 10;
            2'd2:    return 50;
            default: return 90;
        endcase
    endfunction

    function automatic longint wrap_acc(input longint value);
        logic signed [ACC_WIDTH-1:0] truncated;
        truncated = value[ACC_WIDTH-1:0];
        return longint'(truncated);
    endfunction

    task automatic error(input string message);
        n_errors++;
        $display("ERROR cycle %0d: %s", cycle, message);
        if (n_errors >= MAX_ERRORS) end_test();
    endtask

    // -------------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------------
    task automatic fill_random();
        logic [31:0] r;
        for (int k = 0; k < INPUT_SIZE; k++) x_vec[k] = rand_operand();
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < HIDDEN_SIZE; j++) w1_mat[k][j] = rand_operand();
        for (int j = 0; j < HIDDEN_SIZE; j++)
            for (int o = 0; o < OUTPUT_SIZE; o++) w2_mat[j][o] = rand_operand();
        // Biases sized so the hidden values land across the requantizer's range
        for (int j = 0; j < HIDDEN_SIZE; j++) begin
            r = rand32();
            b1_vec[j] = (r[0] ? 1 : -1) * longint'(r[16:1]);
        end
        for (int o = 0; o < OUTPUT_SIZE; o++) begin
            r = rand32();
            b2_vec[o] = (r[0] ? 1 : -1) * longint'(r[16:1]);
        end
    endtask

    task automatic fill_const(input int x_value, input int w1_value,
                              input longint b1_value, input int w2_value, input longint b2_value);
        for (int k = 0; k < INPUT_SIZE; k++) x_vec[k] = x_value;
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < HIDDEN_SIZE; j++) w1_mat[k][j] = w1_value;
        for (int j = 0; j < HIDDEN_SIZE; j++) b1_vec[j] = b1_value;
        for (int j = 0; j < HIDDEN_SIZE; j++)
            for (int o = 0; o < OUTPUT_SIZE; o++) w2_mat[j][o] = w2_value;
        for (int o = 0; o < OUTPUT_SIZE; o++) b2_vec[o] = b2_value;
    endtask

    // Bias values that walk the requantizer through its range, and the
    // activation each one should produce (read back through identity weights).
    // Indexed by a variable so the array bounds hold for any HIDDEN_SIZE.
    // Everything is expressed in steps of 2^SHIFT, so the expected activations
    // below hold for any shift, including 0.
    function automatic longint walk_bias(input int o, input longint step);
        case (o)
            0:       return 0;                        // -> 0
            1:       return step - 1;                 // -> 0, just below one step
            2:       return step;                     // -> 1
            3:       return 127 * step;               // -> 127, the largest that fits
            4:       return 127 * step + (step - 1);  // -> 127
            5:       return 128 * step;               // -> saturates to 127
            6:       return 1000 * step;              // -> saturates to 127
            7:       return -1;                       // -> ReLU 0 -> 0
            8:       return -100000;                  // -> ReLU 0 -> 0
            default: return -1000 * step;             // -> ReLU 0 -> 0
        endcase
    endfunction

    function automatic longint walk_expect(input int o);
        case (o)
            2:             return 1;
            3, 4, 5, 6:    return 127;
            default:       return 0;
        endcase
    endfunction

    // Identity-ish second layer: logit[o] = act[o] + B2[o], so the requantized
    // activations can be read directly at the outputs
    task automatic set_w2_identity();
        for (int j = 0; j < HIDDEN_SIZE; j++)
            for (int o = 0; o < OUTPUT_SIZE; o++) w2_mat[j][o] = (j == o) ? 1 : 0;
    endtask

    // -------------------------------------------------------------------------
    // Reference model: the whole pipeline, and the bias ports
    // -------------------------------------------------------------------------
    task automatic compute_expected();
        longint dot, total, hidden, shifted, logit;
        for (int j = 0; j < HIDDEN_SIZE; j++) begin
            dot = 0;
            for (int k = 0; k < INPUT_SIZE; k++)
                dot += longint'(x_vec[k]) * longint'(w1_mat[k][j]);
            total  = wrap_acc(dot + b1_vec[j]);
            hidden = (total < 0) ? 0 : total;              // ReLU
            if (total < 0) cov_relu_zeroed++;

            shifted = hidden >>> SHIFT;                    // requantize
            if (shifted > ACT_MAX) begin
                act_exp[j] = ACT_MAX;
                cov_act_sat++;
            end else begin
                act_exp[j] = shifted;                      // hidden >= 0, so no low clamp
                if (shifted == 0) cov_act_zero++;
                else              cov_act_mid++;
            end
            bias1_flat[j*ACC_WIDTH +: ACC_WIDTH] = ACC_WIDTH'(b1_vec[j]);
        end

        for (int o = 0; o < OUTPUT_SIZE; o++) begin
            logit = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                logit += act_exp[j] * longint'(w2_mat[j][o]);
            y_exp[o] = wrap_acc(logit + b2_vec[o]);        // no ReLU on the output layer
            if (y_exp[o] > 0) cov_logit_pos++;
            if (y_exp[o] < 0) cov_logit_neg++;
            if (y_exp[o] == 0) cov_logit_zero++;
            bias2_flat[o*ACC_WIDTH +: ACC_WIDTH] = ACC_WIDTH'(b2_vec[o]);
        end
    endtask

    // -------------------------------------------------------------------------
    // Monitor
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
                if (job_in != HOST_BEATS || job_out != OUTPUT_SIZE || job_lasts != 1)
                    error($sformatf("inference had %0d input beats, %0d logits, %0d m_last (expected %0d, %0d, 1)",
                                    job_in, job_out, job_lasts, HOST_BEATS, OUTPUT_SIZE));
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
    // X, then W1 row-major, then W2 row-major
    task automatic send(input int gap_pct, input int n_beats);
        int value, e2;
        bit taken;
        for (int e = 0; e < n_beats; e++) begin
            while (src_chance(gap_pct)) begin
                s_valid = 1'b0;
                @(negedge clk);
            end
            if (e < INPUT_SIZE) begin
                value = x_vec[e];
            end else if (e < L1_BEATS) begin
                e2    = e - INPUT_SIZE;
                value = w1_mat[e2 / HIDDEN_SIZE][e2 % HIDDEN_SIZE];
            end else begin
                e2    = e - L1_BEATS;
                value = w2_mat[e2 / OUTPUT_SIZE][e2 % OUTPUT_SIZE];
            end
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
                        error($sformatf("%s: logit[%0d] = %0d, expected %0d", name, idx, got, y_exp[idx]));
                end
                if (m_last !== (idx == OUTPUT_SIZE - 1))
                    error($sformatf("%s: m_last = %b on logit %0d of %0d", name, m_last, idx, OUTPUT_SIZE));
                idx++;
            end
            @(negedge clk);
        end
        m_ready = 1'b0;
        if (bad > 3) error($sformatf("%s: %0d mismatching logits in total", name, bad));
    endtask

    task automatic log_result(input string name);
        if (results_fd == 0) return;
        n_logged++;
        $fwrite(results_fd, "case %s\nX", name);
        for (int k = 0; k < INPUT_SIZE; k++) $fwrite(results_fd, " %0d", x_vec[k]);
        $fwrite(results_fd, "\nW1");
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < HIDDEN_SIZE; j++) $fwrite(results_fd, " %0d", w1_mat[k][j]);
        $fwrite(results_fd, "\nB1");
        for (int j = 0; j < HIDDEN_SIZE; j++) $fwrite(results_fd, " %0d", b1_vec[j]);
        $fwrite(results_fd, "\nW2");
        for (int j = 0; j < HIDDEN_SIZE; j++)
            for (int o = 0; o < OUTPUT_SIZE; o++) $fwrite(results_fd, " %0d", w2_mat[j][o]);
        $fwrite(results_fd, "\nB2");
        for (int o = 0; o < OUTPUT_SIZE; o++) $fwrite(results_fd, " %0d", b2_vec[o]);
        $fwrite(results_fd, "\nY");
        for (int o = 0; o < OUTPUT_SIZE; o++) $fwrite(results_fd, " %0d", y_got[o]);
        $fwrite(results_fd, "\n");
    endtask

    task automatic run_job(input string name, input int gap_pct, input int stall_pct);
        compute_expected();
        fork
            send(gap_pct, HOST_BEATS);
            receive(name, stall_pct);
        join
        n_checks++;
        if (done !== 1'b1) error($sformatf("%s: done not asserted after the last logit", name));
        log_result(name);
        n_jobs++;
        @(negedge clk);
        n_checks++;
        if (done !== 1'b0 || busy !== 1'b0) error($sformatf("%s: done or busy still high a cycle later", name));
    endtask

    task automatic directed_job(input string name, input bit back_to_back);
        run_job(name, 0, 0);
        n_checks++;
        if (last_busy != BUSY_CYCLES)
            error($sformatf("%s: busy %0d cycles, expected %0d", name, last_busy, BUSY_CYCLES));
        if (back_to_back && last_period != longint'(PERIOD))
            error($sformatf("%s: %0d cycles since the previous inference, expected %0d",
                            name, last_period, PERIOD));
    endtask

    task automatic expect_logits(input longint value, input string name);
        int bad = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) if (y_got[o] != value) bad++;
        n_checks++;
        if (bad != 0) error($sformatf("%s: %0d logits differ from the hand value %0d", name, bad, value));
        else          $display("  pass  %s: all %0d logits = %0d", name, OUTPUT_SIZE, value);
    endtask

    task automatic apply_reset(input string where);
        rst = 1'b1;
        @(negedge clk);
        rst = 1'b0;
        n_checks++;
        if (busy !== 1'b0 || s_ready !== 1'b0 || m_valid !== 1'b0 || done !== 1'b0)
            error($sformatf("after reset during %s: busy=%b s_ready=%b m_valid=%b done=%b",
                            where, busy, s_ready, m_valid, done));
        $display("  pass  reset during %s: back to idle", where);
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

    task automatic cover_bin(input string name, input int unsigned hits);
        $display("  %8d  %s", hits, name);
        if (hits == 0) begin
            n_errors++;
            $display("ERROR coverage hole: '%s' was never exercised", name);
        end
    endtask

    task automatic end_test();
        if (results_fd != 0) begin
            $fwrite(results_fd, "end %0d\n", n_logged);
            $fclose(results_fd);
            results_fd = 0;
        end
        $display("");
        if (n_errors == 0) begin
            $display("TEST PASSED%s: nn_accelerator_tb %0d-%0d-%0d SHIFT=%0d NUM_MACS=%0d seed=%0d | %0d inferences, %0d checks, 0 errors, %0d cycles per inference back to back",
                     run_note, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SHIFT, NUM_MACS, seed, n_jobs, n_checks, PERIOD);
            $finish;
        end else begin
            $display("TEST FAILED%s: nn_accelerator_tb %0d-%0d-%0d SHIFT=%0d seed=%0d | %0d errors in %0d checks",
                     run_note, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SHIFT, seed, n_errors, n_checks);
            $fatal(1, "nn_accelerator_tb failed");
        end
    endtask

    // -------------------------------------------------------------------------
    // Test sections
    // -------------------------------------------------------------------------
    task automatic test_directed();
        longint step;
        section_begin("1. directed, no stalls, back to back");
        step = longint'(1) << SHIFT;

        fill_const(0, 0, 0, 0, 0);
        directed_job("zeros", 1'b0);                       // first inference after reset
        expect_logits(0, "zeros");

        // X = 0, so each hidden value is just its bias. With an identity second
        // layer, logit[o] is exactly the requantized activation of channel o.
        fill_const(0, 0, 0, 0, 0);
        set_w2_identity();
        if (HIDDEN_SIZE >= 10 && OUTPUT_SIZE >= 10) begin
            for (int o = 0; o < 10; o++) b1_vec[o] = walk_bias(o, step);
            directed_job("requantizer walked through its range", 1'b1);
            for (int o = 0; o < 10; o++) begin
                n_checks++;
                if (y_got[o] != walk_expect(o))
                    error($sformatf("requant walk: logit[%0d] = %0d, expected %0d",
                                    o, y_got[o], walk_expect(o)));
            end
            $display("  pass  requantizer through the network: 0, 1, 127 and saturation read back as logits");
        end else begin
            $display("  (requantizer walk needs at least 10 hidden channels and outputs: skipped)");
        end

        // All 127: hidden = INPUT_SIZE*16129 = 258,064, which saturates to 127,
        // so with W2 all ones each logit is HIDDEN_SIZE*127
        fill_const(127, 127, 0, 1, 0);
        directed_job("all 127", 1'b1);
        expect_logits(longint'(HIDDEN_SIZE) * 127, "all 127");

        // Hidden values ReLU zeroes: logits are just B2
        fill_const(-128, 127, 0, 1, 0);
        for (int o = 0; o < OUTPUT_SIZE; o++) b2_vec[o] = longint'(o) * 1000;
        directed_job("ReLU zeroes every hidden value", 1'b1);
        for (int o = 0; o < OUTPUT_SIZE; o++) begin
            n_checks++;
            if (y_got[o] != longint'(o) * 1000)
                error($sformatf("ReLU-zeroed: logit[%0d] = %0d, expected %0d", o, y_got[o], o * 1000));
        end
        $display("  pass  ReLU zeroes every hidden value: logits are the second-layer biases");

        fill_random();
        directed_job("random, no stalls", 1'b1);

        $display("  pass  every inference: busy %0d cycles; back to back %0d cycles apart",
                 BUSY_CYCLES, PERIOD);
        section_end();
    endtask

    task automatic test_protocol();
        section_begin("2. protocol: stalls and resets");

        fill_random();
        run_job("input gaps", 50, 0);
        $display("  pass  input gaps (s_valid low 50%%): busy %0d cycles", last_busy);
        fill_random();
        run_job("output backpressure", 0, 50);
        $display("  pass  output backpressure (m_ready low 50%%): busy %0d cycles", last_busy);
        fill_random();
        run_job("gaps and backpressure", 70, 70);
        $display("  pass  both (70%%): busy %0d cycles", last_busy);

        // Reset at four points in the pipeline
        fill_random();
        compute_expected();
        send(0, L1_BEATS / 2);
        apply_reset("layer 1 load");
        cov_reset_l1_load++;
        fill_random();
        directed_job("after reset during layer 1 load", 1'b0);

        fill_random();
        compute_expected();
        send(0, L1_BEATS);                       // layer 1 computes, layer 2 takes activations
        repeat (INPUT_SIZE) @(negedge clk);
        apply_reset("layer 1 compute");
        cov_reset_l1_calc++;
        fill_random();
        directed_job("after reset during layer 1 compute", 1'b0);

        fill_random();
        compute_expected();
        send(0, L1_BEATS + W2_BEATS / 2);        // partway through W2
        apply_reset("W2 load");
        cov_reset_w2_load++;
        fill_random();
        directed_job("after reset during W2 load", 1'b0);

        fill_random();
        compute_expected();
        m_ready = 1'b0;
        send(0, HOST_BEATS);
        repeat (HIDDEN_SIZE) @(negedge clk);     // layer 2 computing
        apply_reset("layer 2 compute");
        cov_reset_l2_calc++;
        fill_random();
        directed_job("after reset during layer 2 compute", 1'b0);

        section_end();
    endtask

    task automatic test_random();
        logic [31:0] r;
        section_begin($sformatf("3. random (%0d inferences, seed %0d)", N_RANDOM, seed));
        for (int t = 0; t < N_RANDOM; t++) begin
            fill_random();
            r = rand32();
            repeat (int'(r[5:4])) @(negedge clk);
            run_job($sformatf("random %0d", t), stall_rate(r[1:0]), stall_rate(r[3:2]));
        end
        section_end();
    endtask

    task automatic report_coverage();
        section_begin("coverage (whole run)");
        cover_bin("logits > 0",                        cov_logit_pos);
        cover_bin("logits < 0",                        cov_logit_neg);
        cover_bin("logits = 0",                        cov_logit_zero);
        cover_bin("activations saturated at 127",      cov_act_sat);
        cover_bin("activations 0 after requantizing",  cov_act_zero);
        cover_bin("activations between 1 and 126",     cov_act_mid);
        cover_bin("hidden values zeroed by ReLU",      cov_relu_zeroed);
        cover_bin("input gap cycles",                  cov_gap);
        cover_bin("backpressure cycles",               cov_backpressure);
        cover_bin("reset during layer 1 load",         cov_reset_l1_load);
        cover_bin("reset during layer 1 compute",      cov_reset_l1_calc);
        cover_bin("reset during W2 load",              cov_reset_w2_load);
        cover_bin("reset during layer 2 compute",      cov_reset_l2_calc);
    endtask

    // -------------------------------------------------------------------------
    // Main
    // -------------------------------------------------------------------------
    initial begin : main
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        rng_state = (seed == 0) ? 32'h1 : seed;
        rng_src   = xorshift32(rng_state ^ 32'h9E3779B9);
        rng_sink  = xorshift32(rng_state ^ 32'h7F4A7C15);

        if ($value$plusargs("resultsfile=%s", resultsfile)) begin
            results_fd = $fopen(resultsfile, "w");
            if (results_fd == 0) $fatal(1, "nn_accelerator_tb: cannot open %s", resultsfile);
            $fwrite(results_fd, "# nn_accelerator_tb: X, W1, B1, W2, B2 and the logits from the DUT\n");
            $fwrite(results_fd, "dims %0d %0d %0d %0d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SHIFT);
        end
        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, nn_accelerator_tb);
        end

        $display("nn_accelerator_tb: %0d -> %0d (ReLU, >> %0d) -> %0d logits, %0d cycles per inference without stalls",
                 INPUT_SIZE, HIDDEN_SIZE, SHIFT, OUTPUT_SIZE, BUSY_CYCLES);

        rst        = 1'b1;
        s_valid    = 1'b0;
        s_data     = '0;
        m_ready    = 1'b0;
        bias1_flat = '0;
        bias2_flat = '0;
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
        $display("TEST FAILED: nn_accelerator_tb watchdog timeout");
        $fatal(1, "nn_accelerator_tb timeout");
    end

endmodule

`default_nettype wire
