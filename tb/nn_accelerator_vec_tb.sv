// =============================================================================
// nn_accelerator_vec_tb.sv - vector-driven regression harness for the network
// =============================================================================
// Unlike tb/nn_accelerator_tb.sv, this testbench invents nothing. Python writes
// a vector file, this drives it into the DUT, and Python checks what comes out
// against NumPy. See python/run_regression.py.
//
// The vector file is whitespace-separated decimal integers and nothing else
// ($fscanf cannot skip comments), laid out to match the weight-stationary
// interface: weights are loaded once per group and reused by every inference
// in it.
//
//   INPUT_SIZE HIDDEN_SIZE OUTPUT_SIZE SHIFT N_GROUPS
//   per group:
//     N_CASES
//     W1   INPUT_SIZE*HIDDEN_SIZE   row-major
//     B1   HIDDEN_SIZE
//     W2   HIDDEN_SIZE*OUTPUT_SIZE  row-major
//     B2   OUTPUT_SIZE
//     per case:
//       X  INPUT_SIZE
//
// The header must match the parameters this testbench was built with, so a
// mismatched vector file fails loudly instead of being silently misread.
//
// What this testbench checks on its own (everything else is Python's job):
//   * the protocol: m_last only on the final logit, outputs never X/Z,
//     done is a single-cycle pulse after the last logit
//   * a watchdog, so a deadlock fails instead of hanging the regression
// Timing and corner cases are the other testbench's job; this one exists to
// run a large number of Python-chosen cases.
//
// Plusargs:   +vectors=<path>  (required)  +resultsfile=<path>  +seed=<n>
//             +stalls  drives random input gaps and output backpressure
//             +dumpfile=<path>  write a waveform
//             +maxcases=<n>     stop after n cases. Use it with +dumpfile:
//                               $dumpoff is ignored by the simulator, so the
//                               only way to keep a trace small is fewer cases.
// Parameters: INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SHIFT, NUM_MACS
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module nn_accelerator_vec_tb;

    parameter int INPUT_SIZE  = 16;
    parameter int HIDDEN_SIZE = 32;
    parameter int OUTPUT_SIZE = 10;
    parameter int SHIFT       = 8;
    parameter int NUM_MACS    = 1;

    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int CLK_PERIOD = 10;
    localparam int W1_BEATS   = INPUT_SIZE*HIDDEN_SIZE;
    localparam int W2_BEATS   = HIDDEN_SIZE*OUTPUT_SIZE;
    localparam int W_BEATS    = W1_BEATS + W2_BEATS;
    // Generous: one job cannot exceed the weight load plus a full inference,
    // and stalls at most triple it.
    localparam int WATCHDOG   = 8 * (W_BEATS + INPUT_SIZE*HIDDEN_SIZE
                                     + HIDDEN_SIZE*OUTPUT_SIZE + 64);

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic                             clk;
    logic                             rst;
    logic                             load_weights;
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
        .clk          (clk),
        .rst          (rst),
        .load_weights (load_weights),
        .s_valid      (s_valid),
        .s_ready      (s_ready),
        .s_data       (s_data),
        .bias1_flat   (bias1_flat),
        .bias2_flat   (bias2_flat),
        .m_valid      (m_valid),
        .m_ready      (m_ready),
        .m_data       (m_data),
        .m_last       (m_last),
        .busy         (busy),
        .done         (done)
    );

    // Declared here, before the process that drives it: Icarus rejects a
    // forward reference that Verilator accepts.
    longint cycle;                   // free-running, handy as a waveform cursor

    initial begin
        clk   = 1'b0;
        cycle = 0;
    end
    always #(CLK_PERIOD / 2) clk = ~clk;
    always @(posedge clk) cycle <= cycle + 1;

    // -------------------------------------------------------------------------
    // Vectors and bookkeeping
    // -------------------------------------------------------------------------
    int     w1_mat [INPUT_SIZE][HIDDEN_SIZE];
    int     b1_vec [HIDDEN_SIZE];
    int     w2_mat [HIDDEN_SIZE][OUTPUT_SIZE];
    int     b2_vec [OUTPUT_SIZE];
    int     x_vec  [INPUT_SIZE];
    longint y_got  [OUTPUT_SIZE];

    int unsigned n_errors = 0, n_cases = 0, n_groups = 0;
    int          max_cases = 0;          // 0 = no limit
    int          vec_fd = 0, results_fd = 0;
    string       vectors = "", resultsfile = "", dumpfile = "";
    logic [31:0] rng = 32'h1;
    bit          use_stalls = 1'b0;

    function automatic logic [31:0] rand32();
        rng = rng ^ (rng << 13);
        rng = rng ^ (rng >> 17);
        rng = rng ^ (rng << 5);
        return rng;
    endfunction

    // 1 in 4 cycles, only when +stalls was given
    function automatic bit stall();
        return use_stalls && (rand32() % 4 == 0);
    endfunction

    task automatic error(input string message);
        n_errors++;
        $display("ERROR case %0d: %s", n_cases, message);
        if (n_errors >= 10) begin
            $display("");
            $display("TEST FAILED: nn_accelerator_vec_tb: too many errors");
            $fatal(1, "nn_accelerator_vec_tb failed");
        end
    endtask

    // Read one integer, failing loudly at end of file rather than silently
    // leaving the destination unchanged.
    function automatic int read_int(input string what);
        int value;
        int code;
        code = $fscanf(vec_fd, "%d", value);
        if (code != 1) begin
            $display("");
            $display("TEST FAILED: %s: ran out of vectors while reading %s", vectors, what);
            $fatal(1, "nn_accelerator_vec_tb: short vector file");
        end
        return value;
    endfunction

    // -------------------------------------------------------------------------
    // Driving
    // -------------------------------------------------------------------------
    task automatic deadlock(input string where);
        $display("");
        $display("TEST FAILED: nn_accelerator_vec_tb: %s never completed (watchdog)", where);
        $fatal(1, "nn_accelerator_vec_tb deadlock");
    endtask

    // Guarded: a DUT that never raises s_ready must fail the run, not hang it.
    task automatic send_beat(input int value);
        bit taken;
        int guard = 0;
        while (stall()) begin
            s_valid = 1'b0;
            @(negedge clk);
        end
        if (value < -(2**(DATA_WIDTH-1)) || value > 2**(DATA_WIDTH-1) - 1)
            error($sformatf("vector value %0d does not fit in %0d signed bits",
                            value, DATA_WIDTH));
        s_valid = 1'b1;
        s_data  = DATA_WIDTH'(value);
        do begin
            taken = s_ready;
            @(negedge clk);
            guard++;
            if (guard >= WATCHDOG) deadlock("an input beat");
        end while (!taken);
    endtask

    // One weight-load job: W1 then W2, nothing comes back
    task automatic load_group_weights();
        int guard = 0;
        load_weights = 1'b1;
        for (int k = 0; k < INPUT_SIZE; k++)
            for (int j = 0; j < HIDDEN_SIZE; j++) send_beat(w1_mat[k][j]);
        for (int j = 0; j < HIDDEN_SIZE; j++)
            for (int o = 0; o < OUTPUT_SIZE; o++) send_beat(w2_mat[j][o]);
        s_valid      = 1'b0;
        load_weights = 1'b0;                     // already latched; only IDLE samples it
        while (done !== 1'b1 && guard < WATCHDOG) begin
            @(negedge clk);
            guard++;
        end
        if (guard >= WATCHDOG) deadlock("a weight load");
        @(negedge clk);
    endtask

    // One inference: X in, OUTPUT_SIZE logits out
    task automatic run_inference();
        int idx   = 0;
        int guard = 0;
        fork
            begin
                for (int k = 0; k < INPUT_SIZE; k++) send_beat(x_vec[k]);
                s_valid = 1'b0;
            end
            begin
                while (idx < OUTPUT_SIZE && guard < WATCHDOG) begin
                    m_ready = !stall();
                    if (m_valid && m_ready) begin
                        if ($isunknown(m_data))
                            error($sformatf("logit %0d is X or Z", idx));
                        y_got[idx] = longint'(m_data);
                        if (m_last !== (idx == OUTPUT_SIZE - 1))
                            error($sformatf("m_last = %b on logit %0d of %0d",
                                            m_last, idx, OUTPUT_SIZE));
                        idx++;
                    end
                    @(negedge clk);
                    guard++;
                end
                m_ready = 1'b0;
            end
        join
        if (guard >= WATCHDOG) deadlock("an inference");
        if (done !== 1'b1) error("done not asserted after the last logit");
        @(negedge clk);
        if (done !== 1'b0 || busy !== 1'b0) error("done or busy still high a cycle later");
    endtask

    // -------------------------------------------------------------------------
    // Logging, in the format python/verify_results.py already reads
    // -------------------------------------------------------------------------
    task automatic log_case(input int group, input int index);
        if (results_fd == 0) return;
        $fwrite(results_fd, "case group%0d/%0d\nX", group, index);
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

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    int hdr_in, hdr_hid, hdr_out, hdr_shift, hdr_groups, group_cases;

    initial begin : main
        int seed;
        if ($value$plusargs("seed=%d", seed)) rng = (seed == 0) ? 32'h1 : 32'(seed);
        use_stalls = $test$plusargs("stalls");
        if (!$value$plusargs("maxcases=%d", max_cases)) max_cases = 0;
        if ($value$plusargs("dumpfile=%s", dumpfile)) begin
            $dumpfile(dumpfile);
            $dumpvars(0, nn_accelerator_vec_tb);
        end

        if (!$value$plusargs("vectors=%s", vectors))
            $fatal(1, "nn_accelerator_vec_tb: +vectors=<path> is required");
        vec_fd = $fopen(vectors, "r");
        if (vec_fd == 0) $fatal(1, "nn_accelerator_vec_tb: cannot open %s", vectors);

        hdr_in     = read_int("INPUT_SIZE");
        hdr_hid    = read_int("HIDDEN_SIZE");
        hdr_out    = read_int("OUTPUT_SIZE");
        hdr_shift  = read_int("SHIFT");
        hdr_groups = read_int("N_GROUPS");
        if (hdr_in != INPUT_SIZE || hdr_hid != HIDDEN_SIZE || hdr_out != OUTPUT_SIZE
            || hdr_shift != SHIFT) begin
            $display("");
            $display("TEST FAILED: %s is for %0d-%0d-%0d SHIFT=%0d, this build is %0d-%0d-%0d SHIFT=%0d",
                     vectors, hdr_in, hdr_hid, hdr_out, hdr_shift,
                     INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SHIFT);
            $fatal(1, "nn_accelerator_vec_tb: vector file does not match this build");
        end

        if ($value$plusargs("resultsfile=%s", resultsfile)) begin
            results_fd = $fopen(resultsfile, "w");
            if (results_fd == 0) $fatal(1, "nn_accelerator_vec_tb: cannot open %s", resultsfile);
            $fwrite(results_fd, "# nn_accelerator_vec_tb: vectors from %s\n", vectors);
            $fwrite(results_fd, "dims %0d %0d %0d %0d\n",
                    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SHIFT);
        end

        $display("nn_accelerator_vec_tb: %0d-%0d-%0d SHIFT=%0d NUM_MACS=%0d, %0d weight groups from %s%s",
                 INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SHIFT, NUM_MACS, hdr_groups, vectors,
                 use_stalls ? " (with random stalls)" : "");

        rst          = 1'b1;
        load_weights = 1'b0;
        s_valid      = 1'b0;
        s_data       = '0;
        m_ready      = 1'b0;
        bias1_flat   = '0;
        bias2_flat   = '0;
        repeat (2) @(negedge clk);
        rst = 1'b0;

        for (int g = 0; g < hdr_groups; g++) begin
            if (max_cases != 0 && n_cases >= max_cases) break;
            group_cases = read_int("N_CASES");
            for (int k = 0; k < INPUT_SIZE; k++)
                for (int j = 0; j < HIDDEN_SIZE; j++) w1_mat[k][j] = read_int("W1");
            for (int j = 0; j < HIDDEN_SIZE; j++) b1_vec[j] = read_int("B1");
            for (int j = 0; j < HIDDEN_SIZE; j++)
                for (int o = 0; o < OUTPUT_SIZE; o++) w2_mat[j][o] = read_int("W2");
            for (int o = 0; o < OUTPUT_SIZE; o++) b2_vec[o] = read_int("B2");

            for (int j = 0; j < HIDDEN_SIZE; j++)
                bias1_flat[j*ACC_WIDTH +: ACC_WIDTH] = ACC_WIDTH'(b1_vec[j]);
            for (int o = 0; o < OUTPUT_SIZE; o++)
                bias2_flat[o*ACC_WIDTH +: ACC_WIDTH] = ACC_WIDTH'(b2_vec[o]);

            load_group_weights();                 // paid once for the whole group
            n_groups++;

            for (int c = 0; c < group_cases; c++) begin
                if (max_cases != 0 && n_cases >= max_cases) break;
                for (int k = 0; k < INPUT_SIZE; k++) x_vec[k] = read_int("X");
                run_inference();
                log_case(g, c);
                n_cases++;
            end
        end

        $fclose(vec_fd);
        if (results_fd != 0) begin
            $fwrite(results_fd, "end %0d\n", n_cases);
            $fclose(results_fd);
        end

        $display("");
        if (n_errors == 0) begin
            $display("TEST PASSED: nn_accelerator_vec_tb | %0d inferences over %0d weight groups, 0 protocol errors",
                     n_cases, n_groups);
            $finish;
        end else begin
            $display("TEST FAILED: nn_accelerator_vec_tb | %0d protocol errors in %0d inferences",
                     n_errors, n_cases);
            $fatal(1, "nn_accelerator_vec_tb failed");
        end
    end

endmodule

`default_nettype wire
