# =============================================================================
# NN-Accelerator: simulation and verification entry points
# =============================================================================
# Written for GNU Make 3.81 (the version Apple ships with the Xcode CLT).
#
#   make test          lint + every testbench on Verilator (primary simulator)
#   make test-iverilog every testbench on Icarus Verilog (cross-check)
#   make lint          Verilator -Wall lint of every RTL module
#   make mac           MAC testbench, INT8 x INT8 -> INT32 (Verilator)
#   make mac-params    MAC testbench, INT4 x INT4 -> INT12 (parameterization check)
#   make mac-iverilog  both MAC configurations on Icarus Verilog
#   make mac-waves     short MAC run that writes waveforms/mac_tb.vcd
#   make relu          ReLU testbench, WIDTH=32 (Verilator)
#   make relu-params   ReLU testbench, WIDTH=16, exhaustive over all inputs
#   make relu-iverilog both ReLU configurations on Icarus Verilog
#   make relu-waves    short ReLU run that writes waveforms/relu_tb.vcd
#   make ctrl          matrix-multiply controller FSM testbench, 8x8x8 (Verilator)
#   make ctrl-params   controller FSM testbench at 1x1x1 and 2x3x2
#   make ctrl-iverilog all controller configurations on Icarus Verilog
#   make ctrl-waves    short controller run that writes waveforms/matmul_ctrl_tb.vcd
#   make matmul        8x8 matrix multiply testbench + NumPy check (Verilator)
#   make matmul-params matrix multiply at 3x16x5 and 1x16x8, + NumPy check
#   make matmul-iverilog  all matrix multiply configurations on Icarus Verilog
#   make matmul-waves  short matrix multiply run that writes waveforms/matrix_mult_tb.vcd
#   make layer         NN layer testbench, 16->8 + NumPy check (Verilator)
#   make layer-params  NN layer at 4->3 and 32->10, + NumPy check
#   make layer-iverilog  all NN layer configurations on Icarus Verilog
#   make layer-waves   short NN layer run that writes waveforms/nn_layer_tb.vcd
#   make requant       requantizer testbench, INT32 -> INT8 (Verilator)
#   make requant-params  requantizer at 16->4 bits, exhaustive over all inputs
#   make requant-iverilog  both requantizer configurations on Icarus Verilog
#   make net           two-layer network 16-32-10 + NumPy check (Verilator)
#   make net-params    network at 4-3-2, and with SHIFT=0, + NumPy check
#   make net-iverilog  all network configurations on Icarus Verilog
#   make net-waves     short network run that writes waveforms/nn_accelerator_tb.vcd
#   make parallel      measure cycles per inference for NUM_MACS = 1,4,8,16,32
#   make golden        self-test of the NumPy golden model (python/golden_model.py)
#   make venv          create .venv with the Python dependencies (numpy)
#   make check-tools   report which simulators / tools are installed
#   make clean         remove simulation build products
#
#   SEED=<n>           seed for the randomized test sections (default 1)
# =============================================================================

SHELL := /bin/bash

VERILATOR ?= verilator
IVERILOG  ?= iverilog
VVP       ?= vvp
# Use the project virtual environment (make venv) when it exists
PYTHON    ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

SIM_DIR  := sim
WAVE_DIR := waveforms

SEED ?= 1

# ---- Sources ----------------------------------------------------------------
MAC_RTL     := rtl/mac.sv
RELU_RTL    := rtl/relu.sv
CTRL_RTL    := rtl/matmul_pkg.sv rtl/matmul_ctrl.sv
MATMUL_RTL  := rtl/matmul_pkg.sv $(MAC_RTL) rtl/matmul_ctrl.sv rtl/matrix_mult.sv
LAYER_RTL   := $(MATMUL_RTL) $(RELU_RTL) rtl/nn_layer.sv
REQUANT_RTL := rtl/requant.sv
NET_RTL     := $(LAYER_RTL) $(REQUANT_RTL) rtl/nn_accelerator.sv
RTL_SRCS    := rtl/matmul_pkg.sv $(MAC_RTL) $(RELU_RTL) $(REQUANT_RTL) rtl/matmul_ctrl.sv \
               rtl/matrix_mult.sv rtl/nn_layer.sv rtl/nn_accelerator.sv
# Each of these modules is linted as a top level against the full RTL list.
LINT_TOPS   := mac relu requant matmul_ctrl matrix_mult nn_layer nn_accelerator

# ---- Verilator --------------------------------------------------------------
# --binary       build a standalone simulator from an SV testbench (implies --timing)
# --trace        compile in VCD support; a dump is written only with +dumpfile=
# --x-initial unique, together with +verilator+rand+reset+2 at run time, starts
#                every flop at a random value, so a missing reset shows up as a
#                failure instead of hiding behind a convenient 0
# -Wno-unknown-warning-option  Homebrew's Verilator passes a clang warning flag
#                that Apple clang does not know; silence that noise
# --unroll-count 1  keep testbench loops as loops. Verilator otherwise unrolls
#                them into every place a task is used: matrix_mult_tb's main
#                process became one ~80,000-line C++ function (8.7 MB of C++,
#                about 30 s to compile); with loops kept it is 1.2 MB and 5 s.
#                The RTL has no procedural loops, so it is unaffected.
VERILATOR_FLAGS := --binary -j 0 --trace --x-initial unique --quiet --unroll-count 1 \
                   -CFLAGS -Wno-unknown-warning-option
VERILATOR_RUN   := +verilator+rand+reset+2 +verilator+seed+$(SEED)

# ---- Icarus Verilog ---------------------------------------------------------
IVERILOG_FLAGS := -g2012 -Wall

# $(call verilator_run,<tb top>,<sources>,<run name>,<extra build flags>,<extra run args>)
# Builds, runs, logs to sim/<run name>.verilator.log, and fails unless the
# simulator exits 0 AND the log contains the testbench's PASS line.
define verilator_run
	@mkdir -p $(SIM_DIR)/verilator/$(3) $(WAVE_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) $(4) --top-module $(1) --Mdir $(SIM_DIR)/verilator/$(3) $(2)
	set -o pipefail; $(SIM_DIR)/verilator/$(3)/V$(1) $(VERILATOR_RUN) +seed=$(SEED) $(5) \
	    | tee $(SIM_DIR)/$(3).verilator.log
	@grep -q '^TEST PASSED' $(SIM_DIR)/$(3).verilator.log
endef

# $(call iverilog_run,<tb top>,<sources>,<run name>,<extra build flags>,<extra run args>)
define iverilog_run
	@mkdir -p $(SIM_DIR)/iverilog $(WAVE_DIR)
	$(IVERILOG) $(IVERILOG_FLAGS) $(4) -s $(1) -o $(SIM_DIR)/iverilog/$(3).vvp $(2)
	set -o pipefail; $(VVP) -n $(SIM_DIR)/iverilog/$(3).vvp +seed=$(SEED) $(5) \
	    | tee $(SIM_DIR)/$(3).iverilog.log
	@grep -q '^TEST PASSED' $(SIM_DIR)/$(3).iverilog.log
endef

.PHONY: help check-tools lint test test-iverilog clean venv golden parallel \
        mac mac-params mac-iverilog mac-waves \
        relu relu-params relu-iverilog relu-waves \
        ctrl ctrl-params ctrl-iverilog ctrl-waves \
        matmul matmul-params matmul-iverilog matmul-waves \
        layer layer-params layer-iverilog layer-waves \
        requant requant-params requant-iverilog \
        net net-params net-iverilog net-waves

help:
	@grep -E '^#   (make |[A-Z_]+=)' Makefile | sed 's/^#   //'

test: lint golden mac mac-params relu relu-params ctrl ctrl-params matmul matmul-params \
      layer layer-params requant requant-params net net-params
	@echo "== make test: all Verilator checks passed =="

test-iverilog: golden mac-iverilog relu-iverilog ctrl-iverilog matmul-iverilog layer-iverilog \
               requant-iverilog net-iverilog
	@echo "== make test-iverilog: all Icarus checks passed =="

lint:
	@for top in $(LINT_TOPS); do \
	    echo "$(VERILATOR) --lint-only -Wall --quiet --top-module $$top $(RTL_SRCS)"; \
	    $(VERILATOR) --lint-only -Wall --quiet --top-module $$top $(RTL_SRCS) || exit 1; \
	done
	@echo "lint: $(LINT_TOPS) clean under verilator -Wall"

mac:
	$(call verilator_run,mac_tb,$(MAC_RTL) tb/mac_tb.sv,mac_tb,)

mac-params:
	$(call verilator_run,mac_tb,$(MAC_RTL) tb/mac_tb.sv,mac_tb_w4_a12,-GDATA_WIDTH=4 -GACC_WIDTH=12)

mac-iverilog:
	$(call iverilog_run,mac_tb,$(MAC_RTL) tb/mac_tb.sv,mac_tb,)
	$(call iverilog_run,mac_tb,$(MAC_RTL) tb/mac_tb.sv,mac_tb_w4_a12,-Pmac_tb.DATA_WIDTH=4 -Pmac_tb.ACC_WIDTH=12)

# Waveform run: the testbench records the directed sections (reset, products,
# accumulation/control) and skips the bulk sections to keep the VCD small.
mac-waves:
	$(call verilator_run,mac_tb,$(MAC_RTL) tb/mac_tb.sv,mac_tb,,+dumpfile=$(WAVE_DIR)/mac_tb.vcd)
	@echo "wrote $(WAVE_DIR)/mac_tb.vcd -- open it with: surfer $(WAVE_DIR)/mac_tb.vcd"

relu:
	$(call verilator_run,relu_tb,$(RELU_RTL) tb/relu_tb.sv,relu_tb,)

relu-params:
	$(call verilator_run,relu_tb,$(RELU_RTL) tb/relu_tb.sv,relu_tb_w16,-GWIDTH=16)

relu-iverilog:
	$(call iverilog_run,relu_tb,$(RELU_RTL) tb/relu_tb.sv,relu_tb,)
	$(call iverilog_run,relu_tb,$(RELU_RTL) tb/relu_tb.sv,relu_tb_w16,-Prelu_tb.WIDTH=16)

relu-waves:
	$(call verilator_run,relu_tb,$(RELU_RTL) tb/relu_tb.sv,relu_tb,,+dumpfile=$(WAVE_DIR)/relu_tb.vcd)
	@echo "wrote $(WAVE_DIR)/relu_tb.vcd -- open it with: surfer $(WAVE_DIR)/relu_tb.vcd"

# Controller FSM on its own: every output, state and counter checked each cycle
ctrl:
	$(call verilator_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb,)

ctrl-params:
	$(call verilator_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb_1x1x1,-GM=1 -GK=1 -GN=1 -GN_JOBS=400)
	$(call verilator_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb_2x3x2,-GM=2 -GK=3 -GN=2 -GN_JOBS=400)
	$(call verilator_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb_p4,-GNUM_MACS=4)
	$(call verilator_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb_3x16x5_p4,-GM=3 -GK=16 -GN=5 -GNUM_MACS=4 -GN_JOBS=200)

ctrl-iverilog:
	$(call iverilog_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb,)
	$(call iverilog_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb_1x1x1,-Pmatmul_ctrl_tb.M=1 -Pmatmul_ctrl_tb.K=1 -Pmatmul_ctrl_tb.N=1 -Pmatmul_ctrl_tb.N_JOBS=400)
	$(call iverilog_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb_2x3x2,-Pmatmul_ctrl_tb.M=2 -Pmatmul_ctrl_tb.K=3 -Pmatmul_ctrl_tb.N=2 -Pmatmul_ctrl_tb.N_JOBS=400)

ctrl-waves:
	$(call verilator_run,matmul_ctrl_tb,$(CTRL_RTL) tb/matmul_ctrl_tb.sv,matmul_ctrl_tb,,+dumpfile=$(WAVE_DIR)/matmul_ctrl_tb.vcd)
	@echo "wrote $(WAVE_DIR)/matmul_ctrl_tb.vcd -- open it with: surfer $(WAVE_DIR)/matmul_ctrl_tb.vcd"

# Matrix multiply: the testbench checks C against its own reference model and
# also writes every multiply to a results file, which python/verify_results.py
# re-checks against the NumPy golden model.
# $(call matmul_run,<simulator run macro>,<run name>,<build flags>,<simulator>)
define matmul_run
	$(call $(1),matrix_mult_tb,$(MATMUL_RTL) tb/matrix_mult_tb.sv,$(2),$(3),+resultsfile=$(SIM_DIR)/$(2).$(4).results.txt)
	$(PYTHON) python/verify_results.py matmul $(SIM_DIR)/$(2).$(4).results.txt
endef

matmul:
	$(call matmul_run,verilator_run,matrix_mult_tb,,verilator)

matmul-params:
	$(call matmul_run,verilator_run,matrix_mult_tb_3x16x5,-GM=3 -GK=16 -GN=5,verilator)
	$(call matmul_run,verilator_run,matrix_mult_tb_1x16x8,-GM=1 -GK=16 -GN=8,verilator)
	$(call matmul_run,verilator_run,matrix_mult_tb_p4,-GNUM_MACS=4,verilator)
	$(call matmul_run,verilator_run,matrix_mult_tb_3x16x5_p4,-GM=3 -GK=16 -GN=5 -GNUM_MACS=4,verilator)

matmul-iverilog:
	$(call matmul_run,iverilog_run,matrix_mult_tb,,iverilog)
	$(call matmul_run,iverilog_run,matrix_mult_tb_3x16x5,-Pmatrix_mult_tb.M=3 -Pmatrix_mult_tb.K=16 -Pmatrix_mult_tb.N=5,iverilog)
	$(call matmul_run,iverilog_run,matrix_mult_tb_1x16x8,-Pmatrix_mult_tb.M=1 -Pmatrix_mult_tb.K=16 -Pmatrix_mult_tb.N=8,iverilog)

matmul-waves:
	$(call verilator_run,matrix_mult_tb,$(MATMUL_RTL) tb/matrix_mult_tb.sv,matrix_mult_tb,,+dumpfile=$(WAVE_DIR)/matrix_mult_tb.vcd)
	@echo "wrote $(WAVE_DIR)/matrix_mult_tb.vcd -- open it with: surfer $(WAVE_DIR)/matrix_mult_tb.vcd"

# NN layer: same two-reference scheme as the matrix multiply
# $(call layer_run,<simulator run macro>,<run name>,<build flags>,<simulator>)
define layer_run
	$(call $(1),nn_layer_tb,$(LAYER_RTL) tb/nn_layer_tb.sv,$(2),$(3),+resultsfile=$(SIM_DIR)/$(2).$(4).results.txt)
	$(PYTHON) python/verify_results.py layer $(SIM_DIR)/$(2).$(4).results.txt
endef

layer:
	$(call layer_run,verilator_run,nn_layer_tb,,verilator)

layer-params:
	$(call layer_run,verilator_run,nn_layer_tb_4x3,-GINPUT_SIZE=4 -GOUTPUT_SIZE=3,verilator)
	$(call layer_run,verilator_run,nn_layer_tb_32x10,-GINPUT_SIZE=32 -GOUTPUT_SIZE=10,verilator)
	$(call layer_run,verilator_run,nn_layer_tb_norelu,-GAPPLY_RELU=0,verilator)
	$(call layer_run,verilator_run,nn_layer_tb_p8,-GOUTPUT_SIZE=32 -GNUM_MACS=8,verilator)

layer-iverilog:
	$(call layer_run,iverilog_run,nn_layer_tb,,iverilog)
	$(call layer_run,iverilog_run,nn_layer_tb_4x3,-Pnn_layer_tb.INPUT_SIZE=4 -Pnn_layer_tb.OUTPUT_SIZE=3,iverilog)
	$(call layer_run,iverilog_run,nn_layer_tb_32x10,-Pnn_layer_tb.INPUT_SIZE=32 -Pnn_layer_tb.OUTPUT_SIZE=10,iverilog)
	$(call layer_run,iverilog_run,nn_layer_tb_norelu,-Pnn_layer_tb.APPLY_RELU=0,iverilog)

# Requantizer on its own
requant:
	$(call verilator_run,requant_tb,$(REQUANT_RTL) tb/requant_tb.sv,requant_tb,)

requant-params:
	$(call verilator_run,requant_tb,$(REQUANT_RTL) tb/requant_tb.sv,requant_tb_16x4,-GIN_WIDTH=16 -GOUT_WIDTH=4 -GSHIFT=2)

requant-iverilog:
	$(call iverilog_run,requant_tb,$(REQUANT_RTL) tb/requant_tb.sv,requant_tb,)
	$(call iverilog_run,requant_tb,$(REQUANT_RTL) tb/requant_tb.sv,requant_tb_16x4,-Prequant_tb.IN_WIDTH=16 -Prequant_tb.OUT_WIDTH=4 -Prequant_tb.SHIFT=2)

# Two-layer network: testbench plus the NumPy golden model
# $(call net_run,<simulator run macro>,<run name>,<build flags>,<simulator>)
define net_run
	$(call $(1),nn_accelerator_tb,$(NET_RTL) tb/nn_accelerator_tb.sv,$(2),$(3),+resultsfile=$(SIM_DIR)/$(2).$(4).results.txt)
	$(PYTHON) python/verify_results.py network $(SIM_DIR)/$(2).$(4).results.txt
endef

net:
	$(call net_run,verilator_run,nn_accelerator_tb,,verilator)

net-params:
	$(call net_run,verilator_run,nn_accelerator_tb_4x3x2,-GINPUT_SIZE=4 -GHIDDEN_SIZE=3 -GOUTPUT_SIZE=2,verilator)
	$(call net_run,verilator_run,nn_accelerator_tb_shift0,-GSHIFT=0,verilator)
	$(call net_run,verilator_run,nn_accelerator_tb_p8,-GNUM_MACS=8,verilator)
	$(call net_run,verilator_run,nn_accelerator_tb_4x3x2_p4,-GINPUT_SIZE=4 -GHIDDEN_SIZE=3 -GOUTPUT_SIZE=2 -GNUM_MACS=4,verilator)

net-iverilog:
	$(call net_run,iverilog_run,nn_accelerator_tb,,iverilog)
	$(call net_run,iverilog_run,nn_accelerator_tb_4x3x2,-Pnn_accelerator_tb.INPUT_SIZE=4 -Pnn_accelerator_tb.HIDDEN_SIZE=3 -Pnn_accelerator_tb.OUTPUT_SIZE=2,iverilog)
	$(call net_run,iverilog_run,nn_accelerator_tb_shift0,-Pnn_accelerator_tb.SHIFT=0,iverilog)
	$(call net_run,iverilog_run,nn_accelerator_tb_p8,-Pnn_accelerator_tb.NUM_MACS=8,iverilog)

net-waves:
	$(call verilator_run,nn_accelerator_tb,$(NET_RTL) tb/nn_accelerator_tb.sv,nn_accelerator_tb,,+dumpfile=$(WAVE_DIR)/nn_accelerator_tb.vcd)
	@echo "wrote $(WAVE_DIR)/nn_accelerator_tb.vcd -- open it with: surfer $(WAVE_DIR)/nn_accelerator_tb.vcd"

layer-waves:
	$(call verilator_run,nn_layer_tb,$(LAYER_RTL) tb/nn_layer_tb.sv,nn_layer_tb,,+dumpfile=$(WAVE_DIR)/nn_layer_tb.vcd)
	@echo "wrote $(WAVE_DIR)/nn_layer_tb.vcd -- open it with: surfer $(WAVE_DIR)/nn_layer_tb.vcd"

# Phase 7 measurement: cycles per inference of the 16-32-10 network for each
# MAC count. Results come from the testbench's own cycle accounting, which is
# checked against the documented formula on every run.
parallel:
	@rm -f $(SIM_DIR)/parallel.txt
	@for p in 1 4 8 16 32; do \
	    mkdir -p $(SIM_DIR)/verilator/net_p$$p; \
	    $(VERILATOR) $(VERILATOR_FLAGS) -GNUM_MACS=$$p --top-module nn_accelerator_tb \
	        --Mdir $(SIM_DIR)/verilator/net_p$$p $(NET_RTL) tb/nn_accelerator_tb.sv || exit 1; \
	    $(SIM_DIR)/verilator/net_p$$p/Vnn_accelerator_tb $(VERILATOR_RUN) +seed=$(SEED) \
	        > $(SIM_DIR)/parallel_p$$p.log || exit 1; \
	    grep -q '^TEST PASSED' $(SIM_DIR)/parallel_p$$p.log || exit 1; \
	    c=$$(sed -n 's/^TEST PASSED.*0 errors, \([0-9]*\) cycles per inference back to back.*/\1/p' \
	          $(SIM_DIR)/parallel_p$$p.log); \
	    echo "$$p $$c" >> $(SIM_DIR)/parallel.txt; \
	done
	@echo ""
	@echo "16-32-10 network, cycles per inference (simulated; no frequency implied)"
	@awk 'NR==1 {base=$$2} {printf "  %3d MACs  %6d cycles  %5.2fx\n", $$1, $$2, base/$$2}' $(SIM_DIR)/parallel.txt

golden:
	$(PYTHON) python/golden_model.py

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

check-tools:
	@for t in $(VERILATOR) $(IVERILOG) $(VVP) $(PYTHON); do \
	    if command -v $$t >/dev/null 2>&1; then printf '  %-10s %s\n' $$t "$$(command -v $$t)"; \
	    else printf '  %-10s MISSING\n' $$t; fi; \
	done
	@$(VERILATOR) --version 2>/dev/null || true
	@$(IVERILOG) -V 2>/dev/null | head -1 || true
	@$(PYTHON) -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || echo "numpy: MISSING (run make venv)"

clean:
	rm -rf $(SIM_DIR)/verilator $(SIM_DIR)/iverilog $(SIM_DIR)/*.log $(SIM_DIR)/*.txt
	rm -f $(WAVE_DIR)/*.vcd $(WAVE_DIR)/*.fst
