"""Runs all tests.

1. Error tests: tlc.exe on every tests/errors/*.tl. Each file starts with
   '# expect: <text>'. A test passes when tlc.exe exits with code 1 and its
   stderr contains <text>.
2. Correctness tests: every examples/*.tl (except bench_large.tl) is
   compiled, run, and compared with the NumPy reference.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TLC = ROOT / "tlc.exe"
ERRORS_DIR = ROOT / "tests" / "errors"
EXAMPLES_DIR = ROOT / "examples"
TEST_OUT = ROOT / "build" / "test_out"

# bench/ holds runner.py and reference.py
sys.path.insert(0, str(ROOT / "bench"))
import runner  # noqa: E402


def read_expected(tl_path):
    """Returns the text after '# expect: ' on the first line, or None."""
    lines = tl_path.read_text(encoding="utf-8").splitlines()
    prefix = "# expect: "
    if len(lines) == 0 or not lines[0].startswith(prefix):
        return None
    return lines[0][len(prefix):].strip()


def run_error_test(tl_path):
    """Returns (passed, detail)."""
    expected = read_expected(tl_path)
    if expected is None:
        return False, "first line must be '# expect: <text>'"

    result = subprocess.run([str(TLC), str(tl_path)], capture_output=True, text=True)
    message = result.stderr.strip()

    if result.returncode != 1:
        return False, f"exit code {result.returncode}, wanted 1. stderr: {message}"
    if expected not in message:
        return False, f"stderr does not contain '{expected}'. stderr: {message}"
    return True, message


def run_correctness_test(tl_path, flags):
    """Compiles and runs one example, compares with NumPy. Returns (passed, detail)."""
    work_dir = TEST_OUT / tl_path.stem
    if work_dir.exists():
        shutil.rmtree(work_dir)   # no old .out.bin files can make a test pass
    work_dir.mkdir(parents=True)

    c_path = work_dir / f"{tl_path.stem}.c"
    exe_path = work_dir / f"{tl_path.stem}.exe"

    try:
        runner.make_inputs(tl_path, work_dir)
        runner.compile_tl(tl_path, c_path, flags)
        runner.compile_c(c_path, exe_path, "O2")
        runner.run_exe(exe_path, work_dir, 1)
        status, max_err = runner.check(tl_path, work_dir)
    except RuntimeError as error:
        return False, str(error)

    return status == "PASS", f"max_err={max_err:.2e}"


def main():
    if not TLC.exists():
        print(f"{TLC} not found. Run .\\build.bat first.")
        return 1

    passed = 0
    failed = 0

    print("Error tests")
    for tl_path in sorted(ERRORS_DIR.glob("*.tl")):
        ok, detail = run_error_test(tl_path)
        if ok:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        print(f"  {status}  {tl_path.name:<22} {detail}")

    print()
    print("Correctness tests (default flags)")
    for tl_path in sorted(EXAMPLES_DIR.glob("*.tl")):
        if tl_path.name == "bench_large.tl":
            continue
        ok, detail = run_correctness_test(tl_path, [])
        if ok:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        print(f"  {status}  {tl_path.name:<22} {detail}")

    print()
    print(f"{passed} passed, {failed} failed")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
