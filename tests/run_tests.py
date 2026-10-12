"""Runs tlc.exe on every tests/errors/*.tl file.

Each error file starts with '# expect: <text>'. A test passes when tlc.exe
exits with code 1 and its stderr contains <text>.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TLC = ROOT / "tlc.exe"
ERRORS_DIR = ROOT / "tests" / "errors"


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
    print(f"{passed} passed, {failed} failed")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
