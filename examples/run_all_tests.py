"""
Run all unit tests and generate coverage report
"""
import subprocess
import sys


def run_tests():
    """Run pytest with coverage"""
    print("="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)

    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html"
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nCoverage report generated in htmlcov/index.html")
    else:
        print("\n" + "="*80)
        print("❌ SOME TESTS FAILED")
        print("="*80)

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
