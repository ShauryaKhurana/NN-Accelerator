"""Performance summary for the two-layer network, from simulation only.

    perf_report.py [--macs 1,4,8,16,32] [--shape 16,32,10] [--shift N]
                   [--seed S] [--markdown PATH] [--quiet]

Builds and runs tb/nn_accelerator_tb.sv once per MAC width and reports what
the simulation actually measured, plus the figures that follow arithmetically
from those measurements and the network's dimensions.

What is measured (read out of the testbench's own cycle accounting, which it
checks against the documented formula on every run):

    weight load         busy cycles for one load of W1 and W2
    latency             busy cycles for one inference
    interval (waiting)  cycles between inferences when the host waits for done
    interval (streamed) cycles between inferences when the next X is always
                        offered, so layer 1 can start while layer 2 finishes

What is derived (exact arithmetic, not a guess):

    MAC operations      INPUT*HIDDEN + HIDDEN*OUTPUT multiply-accumulates per
                        inference, one per weight
    peak ops/cycle      NUM_MACS, the active array's width
    achieved ops/cycle  MAC operations / streamed interval
    utilization         achieved / peak

One MAC operation here is one multiply-accumulate, not two flops. Each layer
instantiates its own NUM_MACS multipliers, so the network holds 2*NUM_MACS of
them, but only one layer's array is active at a time; utilization is quoted
against the active array and the instantiated total is reported alongside.

These are simulated cycle counts. No clock frequency is implied, and nothing
here is a measurement of an FPGA or ASIC: there are no frequency, power, LUT,
DSP or wall-clock speedup figures, because this design has only ever been
simulated.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = [
    "rtl/matmul_pkg.sv", "rtl/mac.sv", "rtl/relu.sv", "rtl/requant.sv",
    "rtl/matmul_ctrl.sv", "rtl/matrix_mult.sv", "rtl/nn_layer.sv", "rtl/nn_accelerator.sv",
]
TB = "tb/nn_accelerator_tb.sv"
TOP = "nn_accelerator_tb"

# The testbench's own reported numbers
RE_LOAD = re.compile(r"every weight load: busy (\d+) cycles; every inference: (\d+) cycles")
RE_PIPE = re.compile(r"(\d+) cycles apart \((\d+) waiting for done; slower layer alone (\d+)\)")
RE_PASS = re.compile(r"^TEST PASSED.*0 errors, (\d+) cycles per inference back to back", re.M)


class MeasureError(Exception):
    pass


def run_one(shape, shift, num_macs, seed, quiet):
    """Build and run the testbench at one MAC width; return its measurements."""
    in_size, hid, out = shape
    build = ROOT / "sim" / "verilator" / f"perf_p{num_macs}"
    build.mkdir(parents=True, exist_ok=True)
    cmd = ["verilator", "--binary", "-j", "0", "--x-initial", "unique", "--quiet",
           "--unroll-count", "1", "-CFLAGS", "-Wno-unknown-warning-option",
           "--top-module", TOP, "--Mdir", str(build),
           f"-GINPUT_SIZE={in_size}", f"-GHIDDEN_SIZE={hid}", f"-GOUTPUT_SIZE={out}",
           f"-GSHIFT={shift}", f"-GNUM_MACS={num_macs}"] + RTL + [TB]
    if not quiet:
        print(f"  building NUM_MACS={num_macs} ...", flush=True)
    build_proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if build_proc.returncode != 0:
        raise MeasureError(f"build failed at NUM_MACS={num_macs}:\n"
                           + (build_proc.stdout + build_proc.stderr)[-1500:])

    sim = subprocess.run([str(build / f"V{TOP}"), "+verilator+rand+reset+2",
                          f"+verilator+seed+{seed}", f"+seed={seed}"],
                         cwd=ROOT, capture_output=True, text=True)
    text = sim.stdout + sim.stderr
    if sim.returncode != 0 or "TEST PASSED" not in text:
        raise MeasureError(f"simulation failed at NUM_MACS={num_macs}:\n{text[-1500:]}")

    load = RE_LOAD.search(text)
    pipe = RE_PIPE.search(text)
    passline = RE_PASS.search(text)
    if not (load and pipe and passline):
        missing = [n for n, m in (("weight load/latency", load), ("streamed interval", pipe),
                                  ("PASS line", passline)) if not m]
        raise MeasureError(f"could not read {', '.join(missing)} from the NUM_MACS={num_macs} run")

    weight_load, latency = int(load.group(1)), int(load.group(2))
    streamed, waiting, slowest = (int(pipe.group(1)), int(pipe.group(2)), int(pipe.group(3)))
    if waiting != int(passline.group(1)):
        raise MeasureError(f"NUM_MACS={num_macs}: the testbench reported two different "
                           f"intervals, {waiting} and {passline.group(1)}")
    return {"macs": num_macs, "weight_load": weight_load, "latency": latency,
            "waiting": waiting, "streamed": streamed, "slowest_layer": slowest}


def derive(rows, shape):
    """Add the arithmetic that follows from the measurements and the dimensions."""
    in_size, hid, out = shape
    mac_ops = in_size * hid + hid * out          # one multiply-accumulate per weight
    for r in rows:
        r["mac_ops"] = mac_ops
        r["peak"] = r["macs"]                    # active array, one MAC-op per MAC per cycle
        r["achieved"] = mac_ops / r["streamed"]
        r["util"] = r["achieved"] / r["peak"]
        r["speedup"] = rows[0]["streamed"] / r["streamed"]
    return rows


def table(rows, shape, shift):
    in_size, hid, out = shape
    mac_ops = rows[0]["mac_ops"]
    w = []
    w.append(f"Network {in_size}-{hid}-{out}, SHIFT={shift}, INT8 weights and activations")
    w.append(f"{mac_ops} multiply-accumulates per inference "
             f"({in_size}*{hid} + {hid}*{out}), simulated cycles only")
    w.append("")
    w.append("                     measured (cycles)            |        derived")
    w.append("  MACs  weight  latency  interval   interval  slow |   ops/  peak  util   vs")
    w.append("         load             waiting  streamed  layer |  cycle        (%)   1 MAC")
    w.append("  " + "-" * 84)
    for r in rows:
        w.append(f"  {r['macs']:4d}  {r['weight_load']:6d}  {r['latency']:7d}  "
                 f"{r['waiting']:8d}  {r['streamed']:8d}  {r['slowest_layer']:5d} | "
                 f"{r['achieved']:6.2f}  {r['peak']:4d}  {100*r['util']:4.1f}  "
                 f"{r['speedup']:5.2f}x")
    w.append("")
    w.append("  weight load   one load of W1 and W2, reused by every inference after it")
    w.append("  latency       busy cycles for one inference")
    w.append("  interval      cycles between inferences: the host either waits for done,")
    w.append("                or keeps the next X offered so layer 1 starts early")
    w.append("  slow layer    the slower layer's own period; the streamed interval")
    w.append("                cannot go below it")
    w.append("  ops/cycle     mac_ops / streamed interval")
    w.append("  peak          NUM_MACS: the active array's multiply-accumulates per cycle.")
    w.append("                Each layer has its own array, so the network instantiates")
    w.append("                2*NUM_MACS, but only one layer computes at a time.")
    w.append("")
    w.append("  No frequency, power, LUT, DSP or wall-clock figures: this design has only")
    w.append("  been simulated, so none of those would be a measurement.")
    return "\n".join(w)


def markdown(rows, shape, shift):
    in_size, hid, out = shape
    mac_ops = rows[0]["mac_ops"]
    m = []
    m.append("# Performance")
    m.append("")
    m.append(f"Two-layer INT8 network, {in_size}-{hid}-{out}, requantization shift {shift}.")
    m.append(f"One inference is **{mac_ops} multiply-accumulates** "
             f"({in_size}x{hid} + {hid}x{out}, one per weight).")
    m.append("")
    m.append("Generated by `python/perf_report.py` (`make perf`), which builds and runs")
    m.append("`tb/nn_accelerator_tb.sv` at each width and reads the testbench's own cycle")
    m.append("accounting. The testbench checks that accounting against the documented")
    m.append("formula on every run, so these are measurements rather than predictions.")
    m.append("")
    m.append("> These are simulated cycle counts. No clock frequency is implied, and there")
    m.append("> are no FPGA or ASIC figures here — no frequency, power, LUT, DSP or")
    m.append("> wall-clock speedup — because this design has only ever been simulated.")
    m.append("")
    m.append("## Measured")
    m.append("")
    m.append("| MACs | Weight load | Latency | Interval, waiting for `done` | Interval, X always offered | Slower layer alone |")
    m.append("|-----:|------------:|--------:|-----------------------------:|---------------------------:|-------------------:|")
    for r in rows:
        m.append(f"| {r['macs']} | {r['weight_load']:,} | {r['latency']:,} | "
                 f"{r['waiting']:,} | **{r['streamed']:,}** | {r['slowest_layer']:,} |")
    m.append("")
    m.append("*Weight load* is paid once and reused by every inference after it. *Latency*")
    m.append("is the busy window of one inference. The two *interval* columns are the")
    m.append("spacing between inferences: a host that waits for `done` gets the first, one")
    m.append("that keeps the next X offered gets the second, because layer 1 is a separate")
    m.append("FSM and starts while layer 2 is still computing. It cannot beat the slower")
    m.append("layer's own period, shown in the last column.")
    m.append("")
    m.append("## Derived")
    m.append("")
    m.append("| MACs | MAC ops / inference | Achieved ops/cycle | Peak ops/cycle | Utilization | Speedup vs 1 MAC |")
    m.append("|-----:|--------------------:|-------------------:|---------------:|------------:|-----------------:|")
    for r in rows:
        m.append(f"| {r['macs']} | {r['mac_ops']:,} | {r['achieved']:.2f} | {r['peak']} | "
                 f"{100*r['util']:.1f}% | {r['speedup']:.2f}x |")
    m.append("")
    m.append("One MAC operation is one multiply-accumulate, not two flops. Achieved")
    m.append("ops/cycle is the MAC operations per inference divided by the streamed")
    m.append("interval. Peak is `NUM_MACS`, the active array's width: each layer")
    m.append("instantiates its own array, so the network holds `2*NUM_MACS` multipliers,")
    m.append("but only one layer computes at a time, which is why utilization is quoted")
    m.append("against the active array.")
    m.append("")
    best = max(rows, key=lambda r: r["speedup"])
    worst_util = min(rows, key=lambda r: r["util"])
    m.append("## What the numbers say")
    m.append("")
    m.append(f"Going from 1 to {best['macs']} MACs shortens the interval")
    m.append(f"{best['speedup']:.2f}x, from {rows[0]['streamed']:,} cycles to "
             f"{best['streamed']:,}. Utilization falls from "
             f"{100*rows[0]['util']:.1f}% to {100*worst_util['util']:.1f}% over the same")
    m.append("range: the input load, the activation handoff and the output beats do not")
    m.append("shrink with `NUM_MACS`, and the output layer has only")
    m.append(f"{out} columns, so above {out} MACs some of its multipliers have nothing to do.")
    m.append("")
    m.append("The weight load is a flat "
             f"{rows[0]['weight_load']:,} cycles at every width: it is one byte per cycle")
    m.append("through the same port, and more multipliers do not widen it. Because weights")
    m.append("stay resident, a run of *n* inferences costs")
    m.append(f"{rows[0]['weight_load']:,} + {rows[0]['streamed']:,}*n* cycles at one MAC and")
    m.append(f"{best['weight_load']:,} + {best['streamed']:,}*n* at {best['macs']}.")
    m.append("")
    m.append("See [ARCHITECTURE.md](ARCHITECTURE.md) for the schedule these follow from.")
    return "\n".join(m) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--macs", default="1,4,8,16,32", help="MAC widths to compare")
    ap.add_argument("--shape", default="16,32,10", help="INPUT,HIDDEN,OUTPUT")
    ap.add_argument("--shift", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--markdown", nargs="?", const="docs/PERFORMANCE.md", default=None,
                    help="also write a Markdown report (default docs/PERFORMANCE.md)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        widths = [int(v) for v in args.macs.split(",")]
        shape = tuple(int(v) for v in args.shape.split(","))
    except ValueError:
        print("perf_report: --macs and --shape take comma-separated integers", file=sys.stderr)
        return 2
    if len(shape) != 3 or any(v < 1 for v in shape) or any(w < 1 for w in widths):
        print("perf_report: --shape needs three positive integers and --macs positive widths",
              file=sys.stderr)
        return 2

    try:
        rows = [run_one(shape, args.shift, w, args.seed, args.quiet) for w in widths]
    except MeasureError as exc:
        print(f"perf_report: FAIL - {exc}")
        return 1

    derive(rows, shape)
    if not args.quiet:
        print()
    print(table(rows, shape, args.shift))

    if args.markdown:
        path = ROOT / args.markdown
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown(rows, shape, args.shift))
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
