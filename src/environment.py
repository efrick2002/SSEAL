import inspect
import re
from typing import Callable, Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import importlib.util
import sys

class Environment:
    """Abstract base class for environments that LLM agents can interact with."""
    
    def __init__(self, functions_file_path: str):
        """
        Initialize the environment with a Python file containing available functions.
        
        Args:
            functions_file_path (str): Path to the Python file containing functions
        """
        self.functions: Dict[str, Callable] = {}
        self.functions_file_path = functions_file_path
        self.state: Dict[str, Any] = {}
        self._load_functions()
        
    def _load_functions(self) -> None:
        """Load functions from the specified Python file into the environment."""
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("functions_module", self.functions_file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {self.functions_file_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules["functions_module"] = module
            spec.loader.exec_module(module)
            
            # Get all callable attributes that don't start with _
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and not name.startswith('_'):
                    self.functions[name] = obj
                    
        except Exception as e:
            raise EnvironmentError(f"Error loading functions: {str(e)}")
    
    def get_available_functions(self) -> List[str]:
        """
        Get a list of available function names in the environment.
        
        Returns:
            List[str]: List of function names
        """
        return list(self.functions.keys())
    
    def get_function_signature(self, func_name: str) -> str:
        """
        Returns a string containing the function definition with its docstring.

        Args:
            func (Callable): The function to inspect.

        Returns:
            str: A string representation of the function's definition and docstring.
        """
        if func_name not in self.functions:
            raise ValueError(f"Function {func_name} not found in environment")
        
        func = self.functions[func_name]
        signature = inspect.signature(func)
        func_name = func.__name__   
        first_line = f'def {func_name}{signature}:'
        
        doc = inspect.getdoc(func)
        if doc:
            doc_lines = doc.split('\n')
            indented_doc = '    """' + '\n    '.join(doc_lines) + '"""'
            func_def = f"{first_line}\n{indented_doc}"
        else:
            func_def = f"{first_line}\n    pass"
        
        return f"{func_def}\n"
    
    def get_function_context(self) -> str:
        """Get a summary of all available function signiatures in the environment."""

        return '\n'.join([self.get_function_signature(func_name) for func_name in self.functions.keys()])

    def execute_function(self, func_name: str, *args, **kwargs) -> Any:
        """
        Execute a function in the environment.
        
        Args:
            func_name (str): Name of the function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Result of the function execution
        """
        if func_name not in self.functions:
            raise ValueError(f"Function {func_name} not found in environment")
            
        try:
            result = self.functions[func_name](*args, **kwargs)
            self.state['last_function_call'] = {
                'name': func_name,
                'args': args,
                'kwargs': kwargs,
                'result': result
            }
            return result
        except Exception as e:
            raise RuntimeError(f"Error executing function {func_name}: {str(e)}")
    
    def execute_function_string(self, function_str: str) -> Any:
        """
        Execute a function from a string representation like "func(arg1, arg2)".
        
        Args:
            function_str (str): String representation of the function call
            
        Returns:
            Any: Result of the function execution
            
        Example:
            >>> env.execute_function_string("get_user_info(1)")
            {'id': 1, 'name': 'Alice', ...}
        """
        try:
            # Extract function name and arguments
            func_name = function_str.split('(')[0].strip()
            args_str = function_str.split('(')[1].rstrip(')')
            
            # Parse arguments
            args = []
            if args_str:
                # Split by comma, handling potential nested structures
                args = [arg.strip() for arg in args_str.split(',')]
                # Convert string arguments to appropriate types
                args = [int(arg) if arg.isdigit() else arg for arg in args]
            
            # Execute the function
            return self.execute_function(func_name, *args)
            
        except Exception as e:
            raise RuntimeError(f"Error executing function string '{function_str}': {str(e)}")
    
    def execute_function_list(self, function_list_str: str) -> List[Dict[str, Any]]:
        """
        Execute a list of functions from a string containing multiple function calls.
        
        Args:
            function_list_str (str): String containing function calls, possibly within <function_list> tags
            
        Returns:
            List[Dict[str, Any]]: List of results with function calls and their outputs
            
        Example:
            >>> env.execute_function_list('''
                <function_list>
                get_user_info(1)
                get_location_info(2)
                </function_list>
                ''')
            [
                {'call': 'get_user_info(1)', 'result': {'id': 1, 'name': 'Alice', ...}},
                {'call': 'get_location_info(2)', 'result': {'id': 2, 'city': 'Los Angeles', ...}}
            ]
        """
        results = []
        
        # Extract content between <function_list> tags if present
        if '<function_list>' in function_list_str:
            matches = re.findall(r'<function_list>(.*?)</function_list>', function_list_str, re.DOTALL)
            if matches:
                function_list_str = '\n'.join(matches)
        
        # Get individual function calls
        calls = [call.strip() for call in function_list_str.split('\n') if call.strip()]
        
        # Execute each function call
        for call in calls:
            try:
                result = self.execute_function_string(call)
                results.append({
                    'call': call,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'call': call,
                    'error': str(e)
                })
        
        return results
    