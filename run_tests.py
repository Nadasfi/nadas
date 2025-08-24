#!/usr/bin/env python3
"""
Test runner script for Nadas backend
Provides convenient commands to run different test suites
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests"""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term-missing"])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests"""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, "Integration Tests")


def run_e2e_tests(verbose=False):
    """Run end-to-end tests"""
    cmd = ["python", "-m", "pytest", "tests/e2e/"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, "End-to-End Tests")


def run_all_tests(verbose=False, coverage=False):
    """Run all tests"""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term-missing"])
    
    return run_command(cmd, "All Tests")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function"""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, f"Specific Test: {test_path}")


def run_tests_by_marker(marker, verbose=False):
    """Run tests with a specific marker"""
    cmd = ["python", "-m", "pytest", "-m", marker]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, f"Tests with marker: {marker}")


def setup_test_environment():
    """Setup test environment and dependencies"""
    print("Setting up test environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} is installed")
    except ImportError:
        print("❌ pytest not installed. Installing...")
        if not run_command(["pip", "install", "pytest", "pytest-asyncio", "pytest-mock"], "Install pytest"):
            return False
    
    # Check if test directories exist
    test_dirs = ["tests/unit", "tests/integration", "tests/e2e"]
    for test_dir in test_dirs:
        path = Path(test_dir)
        if path.exists():
            print(f"✅ {test_dir} directory exists")
        else:
            print(f"❌ {test_dir} directory missing")
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {test_dir} directory")
    
    print("\n🎯 Test environment ready!")
    return True


def show_test_summary():
    """Show available tests and their descriptions"""
    print("""
📋 Available Test Suites:

🧪 Unit Tests (tests/unit/):
   - test_cross_chain_orchestrator.py: Core orchestrator logic tests
   - test_ai_strategy_generator.py: AI strategy generation tests

🔗 Integration Tests (tests/integration/):
   - test_orchestrator_api.py: API endpoint integration tests

🌐 End-to-End Tests (tests/e2e/):
   - test_cross_chain_workflow.py: Complete workflow tests

📊 Test Markers:
   - unit: Fast unit tests
   - integration: Component interaction tests
   - e2e: End-to-end workflow tests
   - slow: Long-running tests
   - ai: AI-related functionality tests
   - websocket: WebSocket functionality tests

💡 Usage Examples:
   python run_tests.py --unit                    # Run unit tests
   python run_tests.py --integration             # Run integration tests  
   python run_tests.py --e2e                     # Run e2e tests
   python run_tests.py --all                     # Run all tests
   python run_tests.py --marker ai               # Run AI tests
   python run_tests.py --test tests/unit/test_cross_chain_orchestrator.py
   python run_tests.py --coverage                # Run with coverage report
""")


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Test runner for Nadas backend",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Test suite options
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    # Specific test options
    parser.add_argument("--test", type=str, help="Run specific test file or function")
    parser.add_argument("--marker", type=str, help="Run tests with specific marker")
    
    # Configuration options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    # Utility options
    parser.add_argument("--setup", action="store_true", help="Setup test environment")
    parser.add_argument("--summary", action="store_true", help="Show test summary")
    
    args = parser.parse_args()
    
    # Show summary if no arguments or --summary
    if len(sys.argv) == 1 or args.summary:
        show_test_summary()
        return
    
    # Setup environment
    if args.setup:
        setup_test_environment()
        return
    
    # Ensure test environment is ready
    if not setup_test_environment():
        print("❌ Failed to setup test environment")
        sys.exit(1)
    
    # Track test results
    results = []
    
    # Run specific tests
    if args.test:
        results.append(run_specific_test(args.test, args.verbose))
    elif args.marker:
        results.append(run_tests_by_marker(args.marker, args.verbose))
    elif args.unit:
        results.append(run_unit_tests(args.verbose, args.coverage))
    elif args.integration:
        results.append(run_integration_tests(args.verbose))
    elif args.e2e:
        results.append(run_e2e_tests(args.verbose))
    elif args.all:
        results.append(run_all_tests(args.verbose, args.coverage))
    else:
        print("❌ No test suite specified. Use --help for options.")
        sys.exit(1)
    
    # Show results summary
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print('='*60)
    
    if all(results):
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()