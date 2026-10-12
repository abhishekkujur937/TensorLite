"""Benchmark: compiles examples/bench_large.tl with 6 configurations at gcc -O0
and -O3, times each one, checks it against NumPy, and writes bench/results.md.

Method (to avoid order and heat bias on a laptop CPU):
- All 12 programs (6 configs x -O0/-O3) are compiled once, before timing.
- Then 3 rounds. Each round runs all 12 programs once, in the same order.
  One run = 1 untimed warm-up + N timed calls (N = 1 at -O0, 5 at -O3);
  the program reports the median of its N calls.
- A row's time is the median of its 3 round results. The 3 round results are
  also shown, so the noise is visible.

Run from the repo root with the .venv active:
    python bench\\bench.py
For stable numbers: plug in the charger, set power mode to Best performance,
close heavy apps.
"""

import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import runner

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "examples" / "bench_large.tl"
OUT_DIR = ROOT / "bench" / "out"
RESULTS = ROOT / "bench" / "results.md"

# bench_large only: K = 1024 float32 sums need a larger atol (CLAUDE.md Section 15, S11).
ATOL = 1e-3

CONFIGS = [
    ("naive", ["--no-fuse", "--tile=0"]),
    ("fused", ["--tile=0"]),
    ("fused+tile16", ["--tile=16"]),
    ("fused+tile32", ["--tile=32"]),
    ("fused+tile64", ["--tile=64"]),
    ("fused+tile32+unroll4", ["--tile=32", "--unroll=4"]),
]

# (gcc optimization level, timed calls per run)
OPT_LEVELS = [
    ("O0", 1),
    ("O3", 5),
]

ROUNDS = 3


def machine_info():
    """CPU name, L2/L3 size (KB, from Windows), and the gcc version line."""
    command = [
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Processor | Select-Object Name, L2CacheSize, L3CacheSize | ConvertTo-Json",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    cpu = json.loads(result.stdout)
    if isinstance(cpu, list):
        cpu = cpu[0]   # machines with several CPU sockets return a list

    gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
    gcc_line = gcc.stdout.splitlines()[0]

    return {
        "cpu": cpu["Name"].strip(),
        "l2_kb": cpu["L2CacheSize"],
        "l3_kb": cpu["L3CacheSize"],
        "gcc": gcc_line,
    }


def buffer_count(c_path):
    """Reads N from the '/* intermediate buffers: N */' header comment."""
    text = Path(c_path).read_text(encoding="utf-8")
    match = re.search(r"intermediate buffers: (\d+)", text)
    return int(match.group(1))


def build_entry(name, flags, opt, iters):
    """Compiles one config once. Returns a dict that collects its results."""
    file_stem = name.replace("+", "_") + "_" + opt
    c_path = OUT_DIR / f"{file_stem}.c"
    exe_path = OUT_DIR / f"{file_stem}.exe"

    runner.compile_tl(MODEL, c_path, flags)
    runner.compile_c(c_path, exe_path, opt)

    return {
        "name": name,
        "opt": opt,
        "iters": iters,
        "exe_path": exe_path,
        "buffers": buffer_count(c_path),
        "round_ms": [],       # one median per round
        "status": "PASS",     # becomes FAIL if any round fails the check
        "max_err": 0.0,
    }


def run_entry(entry, data_dir):
    """Runs one compiled config once and checks its output against NumPy."""
    # Remove old outputs so a crash can never reuse a previous result.
    for old in data_dir.glob("*.out.bin"):
        old.unlink()

    median_ms = runner.run_exe(entry["exe_path"], data_dir, entry["iters"])
    status, max_err = runner.check(MODEL, data_dir, atol=ATOL)

    entry["round_ms"].append(median_ms)
    if status != "PASS":
        entry["status"] = "FAIL"
    if max_err > entry["max_err"]:
        entry["max_err"] = max_err
    return median_ms, status


def format_table(entries):
    """Builds the Markdown results table. Speedups use the median across rounds."""
    naive_ms = {}
    naive_buffers = 0
    for entry in entries:
        if entry["name"] == "naive":
            naive_ms[entry["opt"]] = entry["median_ms"]
            naive_buffers = entry["buffers"]

    lines = []
    lines.append("| Config | gcc | Calls per round | Per-round ms | Median ms | Speedup vs naive -O0 "
                 "| Speedup vs naive -O3 | Buffers | Buffers eliminated | Correct | Max abs err |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for entry in entries:
        per_round = " / ".join(f"{ms:.0f}" for ms in entry["round_ms"])
        speedup_o0 = naive_ms["O0"] / entry["median_ms"]
        speedup_o3 = naive_ms["O3"] / entry["median_ms"]
        eliminated = naive_buffers - entry["buffers"]
        status = entry["status"]
        if status != "PASS":
            status = "**FAIL**"
        lines.append(
            f"| {entry['name']} | -{entry['opt']} | {entry['iters']} | {per_round} "
            f"| {entry['median_ms']:.2f} | {speedup_o0:.2f}x | {speedup_o3:.2f}x "
            f"| {entry['buffers']} | {eliminated} | {status} | {entry['max_err']:.1e} |"
        )
    return "\n".join(lines)


def main():
    if not runner.TLC.exists():
        print(f"{runner.TLC} not found. Run .\\build.bat first.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = OUT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    runner.make_inputs(MODEL, data_dir)

    info = machine_info()
    print(f"CPU: {info['cpu']}  L2: {info['l2_kb']} KB  L3: {info['l3_kb']} KB")
    print(f"{info['gcc']}")
    print()

    # 1. Compile everything once, before any timing.
    print("compiling 12 programs ...", flush=True)
    entries = []
    for opt, iters in OPT_LEVELS:
        for name, flags in CONFIGS:
            entries.append(build_entry(name, flags, opt, iters))

    # 2. Rounds: every program once per round, always in the same order.
    for round_number in range(1, ROUNDS + 1):
        print(f"round {round_number}/{ROUNDS}", flush=True)
        for entry in entries:
            median_ms, status = run_entry(entry, data_dir)
            print(f"    {entry['name']:<22} -{entry['opt']}  {median_ms:9.2f} ms  {status}", flush=True)

    # 3. A row's time = median of its round results.
    for entry in entries:
        entry["median_ms"] = statistics.median(entry["round_ms"])

    table = format_table(entries)
    print()
    print(table)

    failed = []
    for entry in entries:
        if entry["status"] != "PASS":
            failed.append(f"{entry['name']} -{entry['opt']}")
    if len(failed) > 0:
        print()
        print(f"WARNING: {len(failed)} row(s) FAILED the correctness check: {', '.join(failed)}")

    lines = []
    lines.append("# TensorLite benchmark results")
    lines.append("")
    lines.append(f"Generated by `python bench\\bench.py` on {datetime.now().strftime('%Y-%m-%d %H:%M')}.")
    lines.append("")
    lines.append("## Machine")
    lines.append("")
    lines.append(f"- CPU: {info['cpu']}")
    lines.append("- L1 cache: see Task Manager → Performance → CPU (Windows does not report it reliably)")
    lines.append(f"- L2 cache: {info['l2_kb']} KB (total over all cores, as reported by Windows)")
    lines.append(f"- L3 cache: {info['l3_kb']} KB")
    lines.append(f"- Compiler: {info['gcc']}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Model: `examples\\bench_large.tl` (x[512,1024] → matmul 1024x1024 + bias + relu "
                 "→ matmul 1024x512 + bias + relu)")
    lines.append("- All 12 programs are compiled once before timing.")
    lines.append(f"- {ROUNDS} rounds; each round runs all 12 programs once, in the same order "
                 "(interleaved, so no config always gets the coolest or hottest CPU).")
    lines.append("- One run = 1 untimed warm-up call + N timed calls (N = 1 at -O0, 5 at -O3); "
                 "the program reports the median of its N calls.")
    lines.append(f"- Median ms = median of the {ROUNDS} per-round results.")
    lines.append(f"- Correctness: every run is checked with `np.allclose(out, ref, rtol=1e-4, "
                 f"atol={ATOL:g})` against NumPy (larger atol for K = 1024, see CLAUDE.md Section 15, S11).")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(table)
    lines.append("")
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"wrote {RESULTS}")

    if len(failed) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
