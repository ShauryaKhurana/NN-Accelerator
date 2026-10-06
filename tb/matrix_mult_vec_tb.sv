// =============================================================================
// matrix_mult_vec_tb.sv - vector-driven regression harness for the multiplier
// =============================================================================
// The companion to tb/nn_accelerator_vec_tb.sv, one level down. Python writes
// the vectors, this drives them, Python checks the results against NumPy.
// See python/run_regression.py.
//
// Why both: the network's hidden activations pass through ReLU, so the
// requantizer there never sees a negative value and C is never visible in
// full. Here every element of C leaves the DUT as a raw signed INT32, so
// large negative accumulations are checked directly.
//
// The vector file is whitespace-separated decimal integers and nothing else
// ($fscanf cannot skip comments). B is loaded once per group and reused:
//
//   M K N N_GROUPS
//   per group:
//     N_CASES
//     B   K*N   row-major
//     per case:
//       A   M*K   row-major
//
// The header must match the parameters this testbench was built with.
//
// Checked here: m_last only on the final element, outputs never X/Z, done a
// single-cycle pulse, and a watchdog so a deadlock fails instead of hanging.
// Cycle accounting and corner cases are tb/matrix_mult_tb.sv's job.
//
// Plusargs:   +vectors=<path>  (required)  +resultsfile=<path>  +seed=<n>
//             +stalls  drives random input gaps and output backpressure
// Parameters: M, K, N, NUM_MACS
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module matrix_mult_vec_tb;

    parameter int M        = 8;
    parameter int K        = 8;
    parameter int N        = 8;
    parameter int NUM_MACS = 1;

    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int CLK_PERIOD = 10;
    localparam int WATCHDOG   = 8 * (M*K + K*N + M*N*(K + 1) + 64);

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    logic                         clk;
    logic                         rst;
    logic                         load_weights;
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
        .clk          (clk),
        .rst          (rst),
        .load_weights (load_weights),
        .s_valid      (s_valid),
        .s_ready      (s_ready),
        .s_data       (s_data),
        .m_valid      (m_valid),
        .m_ready      (m_ready),
        .m_data       (m_data),
        .m_last       (m_last),
        .busy         (busy),
        .done         (done)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Vectors and bookkeeping
    // -------------------------------------------------------------------------
    int     a_mat [M][K];
    int     b_mat [K][N];
    longint c_got [M][N];

    int unsigned n_errors = 0, n_cases = 0, n_groups = 0;
    int          vec_fd = 0, results_fd = 0;
    string       vectors = "", resultsfile = "";
    logic [31:0] rng = 32'h1;
    bit          use_stalls = 1'b0;

    function automatic logic [31:0] rand32();
        rng = rng ^ (rng << 13);
        rng = rng ^ (rng >> 17);
        rng = rng ^ (rng << 5);
        return rng;
    endfunction

    function automatic bit stall();
        return use_stalls && (rand32() % 4 == 0);
    endfunction

    task automatic error(input string message);
        n_errors++;
        $display("ERROR case %0d: %s", n_cases, message);
        if (n_errors >= 10) begin
            $display("");
            $display("TEST FAILED: matrix_mult_vec_tb: too many errors");
            $fatal(1, "matrix_mult_vec_tb failed");
        end
    endtask

    function automatic int read_int(input string what);
        int value;
        int code;
        code = $fscanf(vec_fd, "%d", value);
        if (code != 1) begin
            $display("");
            $display("TEST FAILED: %s: ran out of vectors while reading %s", vectors, what);
            $fatal(1, "matrix_mult_vec_tb: short vector file");
        end
        return value;
    endfunction

    task automatic deadlock(input string where);
        $display("");
        $display("TEST FAILED: matrix_mult_vec_tb: %s never completed (watchdog)", where);
        $fatal(1, "matrix_mult_vec_tb deadlock");
    endtask

    // -------------------------------------------------------------------------
    // Driving
    // -------------------------------------------------------------------------
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

    task automatic load_group_weights();
        int guard = 0;
        load_weights = 1'b1;
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++) send_beat(b_mat[k][j]);
        s_valid      = 1'b0;
        load_weights = 1'b0;                     // already latched; only IDLE samples it
        while (done !== 1'b1 && guard < WATCHDOG) begin
            @(negedge clk);
            guard++;
        end
        if (guard >= WATCHDOG) deadlock("a weight load");
        @(negedge clk);
    endtask

    task automatic run_multiply();
        int idx   = 0;
        int guard = 0;
        fork
            begin
                for (int i = 0; i < M; i++)
                    for (int k = 0; k < K; k++) send_beat(a_mat[i][k]);
                s_valid = 1'b0;
            end
            begin
                while (idx < M*N && guard < WATCHDOG) begin
                    m_ready = !stall();
                    if (m_valid && m_ready) begin
                        if ($isunknown(m_data))
                            error($sformatf("C[%0d][%0d] is X or Z", idx / N, idx % N));
                        c_got[idx / N][idx % N] = longint'(m_data);
                        if (m_last !== (idx == M*N - 1))
                            error($sformatf("m_last = %b on element %0d of %0d",
                                            m_last, idx, M*N));
                        idx++;
                    end
                    @(negedge clk);
                    guard++;
                end
                m_ready = 1'b0;
            end
        join
        if (guard >= WATCHDOG) deadlock("a multiply");
        if (done !== 1'b1) error("done not asserted after the last element");
        @(negedge clk);
        if (done !== 1'b0 || busy !== 1'b0) error("done or busy still high a cycle later");
    endtask

    // -------------------------------------------------------------------------
    // Logging, in the format python/verify_results.py already reads
    // -------------------------------------------------------------------------
    task automatic log_case(input int group, input int index);
        if (results_fd == 0) return;
        $fwrite(results_fd, "case group%0d/%0d\nA", group, index);
        for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) $fwrite(results_fd, " %0d", a_mat[i][k]);
        $fwrite(results_fd, "\nB");
        for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) $fwrite(results_fd, " %0d", b_mat[k][j]);
        $fwrite(results_fd, "\nC");
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) $fwrite(results_fd, " %0d", c_got[i][j]);
        $fwrite(results_fd, "\n");
    endtask

    // -------------------------------------------------------------------------
    // Main sequence
    // -------------------------------------------------------------------------
    int hdr_m, hdr_k, hdr_n, hdr_groups, group_cases;

    initial begin : main
        int seed;
        if ($value$plusargs("seed=%d", seed)) rng = (seed == 0) ? 32'h1 : 32'(seed);
        use_stalls = $test$plusargs("stalls");

        if (!$value$plusargs("vectors=%s", vectors))
            $fatal(1, "matrix_mult_vec_tb: +vectors=<path> is required");
        vec_fd = $fopen(vectors, "r");
        if (vec_fd == 0) $fatal(1, "matrix_mult_vec_tb: cannot open %s", vectors);

        hdr_m      = read_int("M");
        hdr_k      = read_int("K");
        hdr_n      = read_int("N");
        hdr_groups = read_int("N_GROUPS");
        if (hdr_m != M || hdr_k != K || hdr_n != N) begin
            $display("");
            $display("TEST FAILED: %s is for %0dx%0dx%0d, this build is %0dx%0dx%0d",
                     vectors, hdr_m, hdr_k, hdr_n, M, K, N);
            $fatal(1, "matrix_mult_vec_tb: vector file does not match this build");
        end

        if ($value$plusargs("resultsfile=%s", resultsfile)) begin
            results_fd = $fopen(resultsfile, "w");
            if (results_fd == 0) $fatal(1, "matrix_mult_vec_tb: cannot open %s", resultsfile);
            $fwrite(results_fd, "# matrix_mult_vec_tb: vectors from %s\n", vectors);
            $fwrite(results_fd, "dims %0d %0d %0d\n", M, K, N);
        end

        $display("matrix_mult_vec_tb: C[%0dx%0d] = A[%0dx%0d] x B[%0dx%0d] NUM_MACS=%0d, %0d weight groups from %s%s",
                 M, N, M, K, K, N, NUM_MACS, hdr_groups, vectors,
                 use_stalls ? " (with random stalls)" : "");

        rst          = 1'b1;
        load_weights = 1'b0;
        s_valid      = 1'b0;
        s_data       = '0;
        m_ready      = 1'b0;
        repeat (2) @(negedge clk);
        rst = 1'b0;

        for (int g = 0; g < hdr_groups; g++) begin
            group_cases = read_int("N_CASES");
            for (int k = 0; k < K; k++)
                for (int j = 0; j < N; j++) b_mat[k][j] = read_int("B");

            load_group_weights();                 // paid once for the whole group
            n_groups++;

            for (int c = 0; c < group_cases; c++) begin
                for (int i = 0; i < M; i++)
                    for (int k = 0; k < K; k++) a_mat[i][k] = read_int("A");
                run_multiply();
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
            $display("TEST PASSED: matrix_mult_vec_tb | %0d multiplies over %0d weight groups, 0 protocol errors",
                     n_cases, n_groups);
            $finish;
        end else begin
            $display("TEST FAILED: matrix_mult_vec_tb | %0d protocol errors in %0d multiplies",
                     n_errors, n_cases);
            $fatal(1, "matrix_mult_vec_tb failed");
        end
    end

endmodule

`default_nettype wire
