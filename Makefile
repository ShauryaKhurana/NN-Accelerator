# =============================================================================
# NN-Accelerator: simulation and verification entry points
# =============================================================================
# Written for GNU Make 3.81 (the version Apple ships with the Xcode CLT).
#
#   make check-tools   report which simulators / tools are installed
#   make clean         remove simulation build products
# =============================================================================

SHELL := /bin/bash

VERILATOR ?= verilator
IVERILOG  ?= iverilog
VVP       ?= vvp
PYTHON    ?= python3

SIM_DIR  := sim
WAVE_DIR := waveforms

.PHONY: help check-tools clean

help:
	@grep -E '^#   make ' Makefile | sed 's/^#   //'

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
