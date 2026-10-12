"""Shared helpers for tests and benchmarks: make inputs, compile, run, compare."""

import subprocess
from pathlib import Path

import numpy as np

import reference

ROOT = Path(__file__).resolve().parent.parent
TLC = ROOT / "tlc.exe"


def make_inputs(tl_path, data_dir, seed=0):
    """Writes random float32 values in [-1, 1] for every input and weight
    as <data_dir>/<name>.bin."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for stmt in reference.parse_tl(tl_path):
        if stmt["kind"] != "decl":
            continue
        values = rng.uniform(-1.0, 1.0, stmt["shape"]).astype(np.float32)
        values.tofile(data_dir / f"{stmt['name']}.bin")


def compile_tl(tl_path, c_path, flags):
    """Runs tlc.exe <tl_path> -o <c_path> <flags>. Raises RuntimeError on failure."""
    command = [str(TLC), str(tl_path), "-o", str(c_path)] + list(flags)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"tlc failed: {result.stderr.strip()}")


def compile_c(c_path, exe_path, opt):
    """Runs gcc -<opt> -o <exe_path> <c_path>. Raises RuntimeError on failure."""
    command = ["gcc", f"-{opt}", "-o", str(exe_path), str(c_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcc failed: {result.stderr.strip()}")


def run_exe(exe_path, data_dir, iters):
    """Runs the generated program and returns median_ms as a float."""
    command = [str(exe_path), str(data_dir), str(iters)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{Path(exe_path).name} failed: {result.stderr.strip()}")

    for line in result.stdout.splitlines():
        if line.startswith("median_ms="):
            return float(line[len("median_ms="):])
    raise RuntimeError(f"no median_ms in output: {result.stdout.strip()}")


def check(tl_path, data_dir, atol=1e-4):
    """Compares every <name>.out.bin with the NumPy reference using
    np.allclose(rtol=1e-4, atol=atol). atol is 1e-4 everywhere except
    bench_large.tl in bench.py (CLAUDE.md Section 15, S11).
    Returns ("PASS" or "FAIL", max abs error)."""
    data_dir = Path(data_dir)
    expected = reference.run_reference(tl_path, data_dir)

    status = "PASS"
    max_err = 0.0
    for name, ref in expected.items():
        out_path = data_dir / f"{name}.out.bin"
        if not out_path.exists():
            return "FAIL", float("inf")

        got = np.fromfile(out_path, dtype=np.float32)
        if got.size != ref.size:
            return "FAIL", float("inf")
        got = got.reshape(ref.shape)

        err = float(np.max(np.abs(got - ref)))
        if err > max_err:
            max_err = err
        if not np.allclose(got, ref, rtol=1e-4, atol=atol):
            status = "FAIL"

    return status, max_err
