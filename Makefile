# =============================================================================
# NN-Accelerator: simulation and verification entry points
# =============================================================================
# Written for GNU Make 3.81 (the version Apple ships with the Xcode CLT).
#
#   make test          lint + every testbench on Verilator (primary simulator)
#   make test-iverilog every testbench on Icarus Verilog (cross-check)
#   make lint          Verilator -Wall lint of the RTL
#   make mac           MAC testbench, INT8 x INT8 -> INT32 (Verilator)
#   make mac-params    MAC testbench, INT4 x INT4 -> INT12 (parameterization check)
#   make mac-iverilog  both MAC configurations on Icarus Verilog
#   make mac-waves     short MAC run that writes waveforms/mac_tb.vcd
#   make check-tools   report which simulators / tools are installed
#   make clean         remove simulation build products
#
#   SEED=<n>           seed for the randomized test sections (default 1)
# =============================================================================

SHELL := /bin/bash

VERILATOR ?= verilator
IVERILOG  ?= iverilog
VVP       ?= vvp
PYTHON    ?= python3

SIM_DIR  := sim
WAVE_DIR := waveforms

SEED ?= 1

# ---- Sources ----------------------------------------------------------------
MAC_RTL := rtl/mac.sv

# ---- Verilator --------------------------------------------------------------
# --binary       build a standalone simulator from an SV testbench (implies --timing)
# --trace        compile in VCD support; a dump is written only with +dumpfile=
# --x-initial unique, together with +verilator+rand+reset+2 at run time, starts
#                every flop at a random value, so a missing reset shows up as a
#                failure instead of hiding behind a convenient 0
# -Wno-unknown-warning-option  Homebrew's Verilator passes a clang warning flag
#                that Apple clang does not know; silence that noise
VERILATOR_FLAGS := --binary -j 0 --trace --x-initial unique --quiet \
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

.PHONY: help check-tools lint test test-iverilog mac mac-params mac-iverilog mac-waves clean

help:
	@grep -E '^#   (make |[A-Z]+=)' Makefile | sed 's/^#   //'

test: lint mac mac-params
	@echo "== make test: all Verilator checks passed =="

test-iverilog: mac-iverilog
	@echo "== make test-iverilog: all Icarus checks passed =="

lint:
	$(VERILATOR) --lint-only -Wall --quiet --top-module mac $(MAC_RTL)
	@echo "lint: rtl/mac.sv clean under verilator -Wall"

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

check-tools:
	@for t in $(VERILATOR) $(IVERILOG) $(VVP) $(PYTHON); do \
	    if command -v $$t >/dev/null 2>&1; then printf '  %-10s %s\n' $$t "$$(command -v $$t)"; \
	    else printf '  %-10s MISSING\n' $$t; fi; \
	done
	@$(VERILATOR) --version 2>/dev/null || true
	@$(IVERILOG) -V 2>/dev/null | head -1 || true

clean:
	rm -rf $(SIM_DIR)/verilator $(SIM_DIR)/iverilog $(SIM_DIR)/*.log
	rm -f $(WAVE_DIR)/*.vcd $(WAVE_DIR)/*.fst
