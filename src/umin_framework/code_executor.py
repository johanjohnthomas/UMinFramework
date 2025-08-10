"""
Code execution sandbox for safely running and testing generated code.

This module provides secure code execution capabilities for benchmarking
LLM-generated code solutions, with support for timeouts, resource limits,
and pass@k metric calculation.
"""

import ast
import contextlib
import io
import multiprocessing
import os
import signal
import sys
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
import tempfile
import subprocess


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    timeout: bool = False
    memory_usage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'output': self.output,
            'error': self.error,
            'execution_time': self.execution_time,
            'timeout': self.timeout,
            'memory_usage': self.memory_usage
        }


class CodeValidator:
    """Validates code for safety before execution."""
    
    # Dangerous operations to block
    DANGEROUS_IMPORTS = {
        'os', 'sys', 'subprocess', 'importlib', '__import__',
        'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input',
        'reload', 'quit', 'exit', 'help', 'copyright', 'credits', 'license'
    }
    
    DANGEROUS_ATTRIBUTES = {
        '__import__', '__loader__', '__spec__', '__package__', '__file__',
        '__cached__', '__builtins__'
    }
    
    def __init__(self, allow_imports: Optional[List[str]] = None):
        """
        Initialize the code validator.
        
        Args:
            allow_imports: List of allowed import modules
        """
        self.allowed_imports = set(allow_imports or [])
        # Add commonly needed safe modules
        self.allowed_imports.update({
            'math', 'random', 'itertools', 'collections', 'functools',
            'operator', 'copy', 'json', 're', 'string', 'datetime',
            'decimal', 'fractions', 'statistics'
        })
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate code for safety.
        
        Args:
            code: Code string to validate
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            # Parse the code into AST
            tree = ast.parse(code)
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name in self.DANGEROUS_IMPORTS:
                            return False, f"Dangerous import: {module_name}"
                        if module_name not in self.allowed_imports:
                            return False, f"Disallowed import: {module_name}"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name in self.DANGEROUS_IMPORTS:
                            return False, f"Dangerous import from: {module_name}"
                        if module_name not in self.allowed_imports:
                            return False, f"Disallowed import from: {module_name}"
                
                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.DANGEROUS_IMPORTS:
                            return False, f"Dangerous function call: {func_name}"
                
                # Check for dangerous attribute access
                elif isinstance(node, ast.Attribute):
                    if node.attr in self.DANGEROUS_ATTRIBUTES:
                        return False, f"Dangerous attribute access: {node.attr}"
                
                # Block while loops that might be infinite
                elif isinstance(node, ast.While):
                    # Allow only simple conditions that are likely to terminate
                    if isinstance(node.test, ast.NameConstant) and node.test.value is True:
                        return False, "Infinite while loop detected (while True:)"
                    elif isinstance(node.test, ast.Constant) and node.test.value is True:
                        return False, "Infinite while loop detected (while True:)"
            
            return True, ""
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"


class SafeCodeExecutor:
    """Safe code executor with sandboxing and resource limits."""
    
    def __init__(
        self,
        timeout: float = 30.0,
        memory_limit: Optional[int] = None,
        validator: Optional[CodeValidator] = None
    ):
        """
        Initialize the safe code executor.
        
        Args:
            timeout: Maximum execution time in seconds
            memory_limit: Maximum memory usage in bytes (not implemented)
            validator: Code validator instance
        """
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.validator = validator or CodeValidator()
    
    def _execute_in_sandbox(self, code: str, test_code: str = "") -> ExecutionResult:
        """
        Execute code in a sandboxed environment.
        
        Args:
            code: Main code to execute
            test_code: Test code to run after main code
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        try:
            # Create restricted globals
            safe_globals = {
                '__builtins__': {
                    # Safe built-in functions
                    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                    'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
                    'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
                    'hex': hex, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
                    'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max,
                    'min': min, 'next': next, 'oct': oct, 'ord': ord, 'pow': pow,
                    'print': print, 'range': range, 'reversed': reversed, 'round': round,
                    'set': set, 'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum,
                    'tuple': tuple, 'type': type, 'zip': zip,
                    # Constants
                    'True': True, 'False': False, 'None': None,
                    # Exceptions
                    'Exception': Exception, 'AssertionError': AssertionError,
                    'ValueError': ValueError, 'TypeError': TypeError, 'IndexError': IndexError,
                    'KeyError': KeyError, 'AttributeError': AttributeError,
                }
            }
            safe_locals = {}
            
            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                
                # Execute main code
                exec(code, safe_globals, safe_locals)
                
                # Execute test code if provided
                if test_code.strip():
                    exec(test_code, safe_globals, safe_locals)
                
                execution_time = time.time() - start_time
                
                # Get captured output
                stdout_content = stdout_capture.getvalue()
                stderr_content = stderr_capture.getvalue()
                
                return ExecutionResult(
                    success=True,
                    output=stdout_content,
                    error=stderr_content,
                    execution_time=execution_time
                )
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except AssertionError as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error=f"Test assertion failed: {e}",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error=f"Execution error: {e}",
                execution_time=execution_time
            )
    
    def execute(self, code: str, test_code: str = "") -> ExecutionResult:
        """
        Execute code with safety validation and timeout.
        
        Args:
            code: Main code to execute
            test_code: Test code to run
            
        Returns:
            ExecutionResult with execution details
        """
        # Validate code first
        is_safe, validation_error = self.validator.validate_code(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                error=f"Code validation failed: {validation_error}"
            )
        
        # Validate test code if provided
        if test_code.strip():
            is_safe, validation_error = self.validator.validate_code(test_code)
            if not is_safe:
                return ExecutionResult(
                    success=False,
                    error=f"Test code validation failed: {validation_error}"
                )
        
        # Execute with timeout using multiprocessing
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._execute_in_sandbox, code, test_code)
                result = future.result(timeout=self.timeout)
                return result
                
        except FutureTimeoutError:
            return ExecutionResult(
                success=False,
                error="Execution timeout",
                timeout=True,
                execution_time=self.timeout
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Executor error: {e}"
            )


class PassAtKCalculator:
    """Calculator for pass@k metrics."""
    
    @staticmethod
    def calculate_pass_at_k(results: List[bool], k: int) -> float:
        """
        Calculate pass@k metric.
        
        The pass@k metric measures the probability that at least one of the
        top k generated solutions is correct.
        
        Args:
            results: List of boolean results (True for passed tests)
            k: Number of samples to consider
            
        Returns:
            Pass@k score between 0 and 1
        """
        if not results or k <= 0:
            return 0.0
        
        n = len(results)
        if k >= n:
            # If k >= n, pass@k is just the proportion of passed tests
            return sum(results) / n
        
        # For k < n, we need to calculate the probability that at least one
        # of k randomly selected samples passes
        passed = sum(results)
        failed = n - passed
        
        if passed == 0:
            return 0.0
        if failed == 0:
            return 1.0
        
        # Use combinatorial formula: 1 - C(failed, k) / C(total, k)
        # This calculates the probability of NOT getting all failures
        from math import comb
        
        try:
            total_combinations = comb(n, k)
            failure_combinations = comb(failed, k) if k <= failed else 0
            return 1.0 - (failure_combinations / total_combinations)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def calculate_multiple_k(results: List[bool], k_values: List[int]) -> Dict[int, float]:
        """
        Calculate pass@k for multiple k values.
        
        Args:
            results: List of boolean results
            k_values: List of k values to calculate
            
        Returns:
            Dictionary mapping k to pass@k score
        """
        return {
            k: PassAtKCalculator.calculate_pass_at_k(results, k)
            for k in k_values
        }
    
    @staticmethod
    def calculate_by_problem(
        problem_results: Dict[str, List[bool]], 
        k_values: List[int]
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate pass@k metrics grouped by problem.
        
        Args:
            problem_results: Dictionary mapping problem_id to list of results
            k_values: List of k values to calculate
            
        Returns:
            Dictionary mapping problem_id to pass@k scores
        """
        return {
            problem_id: PassAtKCalculator.calculate_multiple_k(results, k_values)
            for problem_id, results in problem_results.items()
        }


# Convenience function for simple execution
def execute_code_safely(
    code: str,
    test_code: str = "",
    timeout: float = 30.0,
    allow_imports: Optional[List[str]] = None
) -> ExecutionResult:
    """
    Convenience function to execute code safely.
    
    Args:
        code: Code to execute
        test_code: Test code to run
        timeout: Execution timeout
        allow_imports: List of allowed imports
        
    Returns:
        ExecutionResult
    """
    validator = CodeValidator(allow_imports=allow_imports)
    executor = SafeCodeExecutor(timeout=timeout, validator=validator)
    return executor.execute(code, test_code)


# Example usage
if __name__ == "__main__":
    # Test the code executor
    test_code = '''
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
'''
    
    test_cases = '''
assert add(2, 3) == 5
assert multiply(4, 5) == 20
print("All tests passed!")
'''
    
    print("Testing safe code execution...")
    result = execute_code_safely(test_code, test_cases, timeout=5.0)
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")
    print(f"Time: {result.execution_time:.3f}s")
    
    # Test dangerous code
    dangerous_code = '''
import os
os.system("rm -rf /")
'''
    
    print("\nTesting dangerous code detection...")
    result = execute_code_safely(dangerous_code)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    
    # Test pass@k calculation
    print("\nTesting pass@k calculation...")
    test_results = [True, False, True, True, False, True, False, True]
    
    for k in [1, 2, 3, 5]:
        score = PassAtKCalculator.calculate_pass_at_k(test_results, k)
        print(f"Pass@{k}: {score:.3f}")