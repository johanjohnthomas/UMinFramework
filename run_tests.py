#!/usr/bin/env python3
"""
Test runner script for UMinFramework.

This script runs all unit tests and provides a summary of the results.
Can be used during development to verify all components work correctly.
"""

import sys
import unittest
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def run_all_tests():
    """Run all unit tests and return the results."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(str(start_dir), pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
            
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if result.skipped:
        print(f"\nSKIPPED ({len(result.skipped)}):")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n✓ All {result.testsRun} tests passed!")
    else:
        print(f"\n✗ {len(result.failures) + len(result.errors)} test(s) failed!")
    
    print("=" * 70)
    return success

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)