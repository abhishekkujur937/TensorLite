"""Checks whether "fused is faster than naive at -O3" on bench_large.tl is real.

Runs naive -O3 and fused -O3 alternately (naive, fused, naive, fused, ...),
3 times each, and prints all 6 times. Alternating means both programs see the
same CPU conditions (heat, power mode, core type).

Run from the repo root with the .venv active:
    python bench\\fusion_check.py
"""

import statistics
import sys
from pathlib import Path

import runner

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "examples" / "bench_large.tl"
OUT_DIR = ROOT / "bench" / "out" / "fusion_check"

ROUNDS = 3
ITERS = 5   # timed calls per run, same as -O3 in bench.py

PROGRAMS = [
    ("naive", ["--no-fuse", "--tile=0"]),
    ("fused", ["--tile=0"]),
]


def main():
    if not runner.TLC.exists():
        print(f"{runner.TLC} not found. Run .\\build.bat first.")
        return 1

    data_dir = OUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    runner.make_inputs(MODEL, data_dir)

    # Compile both once.
    exe_paths = {}
    for name, flags in PROGRAMS:
        c_path = OUT_DIR / f"{name}_O3.c"
        exe_path = OUT_DIR / f"{name}_O3.exe"
        runner.compile_tl(MODEL, c_path, flags)
        runner.compile_c(c_path, exe_path, "O3")
        exe_paths[name] = exe_path

    # Alternate: naive, fused, naive, fused, naive, fused.
    times = {"naive": [], "fused": []}
    for round_number in range(1, ROUNDS + 1):
        for name, _ in PROGRAMS:
            median_ms = runner.run_exe(exe_paths[name], data_dir, ITERS)
            status, _ = runner.check(MODEL, data_dir, atol=1e-3)
            times[name].append(median_ms)
            print(f"round {round_number}  {name:<6} -O3  {median_ms:9.2f} ms  {status}", flush=True)

    naive_median = statistics.median(times["naive"])
    fused_median = statistics.median(times["fused"])
    print()
    print(f"naive median: {naive_median:.2f} ms")
    print(f"fused median: {fused_median:.2f} ms")
    print(f"naive / fused: {naive_median / fused_median:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
