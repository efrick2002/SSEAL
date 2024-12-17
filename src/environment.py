import inspect
import re
from typing import Callable, Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import importlib.util
import sys
import ast
from typing import Any, Callable, Dict, List
from copy import deepcopy

def safer_eval(input, context):
    try:
        return eval(input, context)
    except Exception as e:
        return e

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

        # Create a dictionary to store the global context
        self.global_context = {}

        # Execute the file in the global context
        with open(functions_file_path, "r") as file:
            exec(file.read(), self.global_context)

        self.functions_file_path = functions_file_path
        
    def reset_original_context(self):
        self.global_context = {}
        with open(self.functions_file_path, "r") as file:
            exec(file.read(), self.global_context)

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
            func_def = f"{first_line}\n"
        
        return f"{func_def}\n   pass\n"
    
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
    

    # Assuming FunctionExecutor is already defined as per the previous assistant's response

    def parse_function_calls(self, multi_line_string: str) -> list:
        """
        Parses a multi-line string containing multiple function calls (possibly spanning multiple lines)
        into a list of individual function call strings.

        :param multi_line_string: The input string containing function calls separated by newlines.
        :return: A list of function call strings.
        """
        function_calls = []
        current_call = []
        paren_balance = 0
        in_string = False
        string_char = ''

        for line_number, line in enumerate(multi_line_string.splitlines(), start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue  # Skip empty lines

            if stripped_line.startswith("#"):
                continue # Skip comment lines

            for i, char in enumerate(line):
                if in_string:
                    if char == string_char:
                        # Check for escaped quote
                        if i > 0 and line[i - 1] != '\\':
                            in_string = False
                else:
                    if char in ('"', "'"):
                        in_string = True
                        string_char = char
                    elif char == '(':
                        paren_balance += 1
                    elif char == ')':
                        paren_balance -= 1
                        if paren_balance < 0:
                            raise ValueError(f"Unbalanced parentheses at line {line_number}. Input was:\n{multi_line_string}")
                # Ignore other characters

            current_call.append(line)

            if paren_balance == 0 and current_call:
                # Function call is complete
                call_str = '\n'.join(current_call).strip()
                if call_str:
                    function_calls.append(call_str)
                current_call = []

        if paren_balance != 0:
            raise ValueError("Unbalanced parentheses in the input string.")

        # Add any residual function call
        if current_call:
            call_str = '\n'.join(current_call).strip()
            if call_str:
                function_calls.append(call_str)
        
        print(multi_line_string, file=open("debug_fc.txt", "a"))

        for call in function_calls:
            print(call, file=open("debug_fc.txt", "a"))
            print("\n\n", file=open("debug_fc.txt", "a"))

        print("\n\n" + "=" * 100 + "\n\n", file=open("debug_fc.txt", "a"))

        return function_calls
    
    def parse_call(self, code_string):
        """
        Parses a string into an AST and extracts all function call names.

        Args:
            code_string: The string containing Python code.

        Returns:
            A list of function call names (strings) found in the code.
            Returns an empty list if parsing fails or no calls are found.
        """
        try:
            tree = ast.parse(code_string)
            calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            return calls
        except SyntaxError as e:
            raise e(f"Warning: Syntax error encountered while parsing: {code_string}")
        
    def validate_function_call(self, code_string):
        """
        Validates that all function calls in a string are from an approved list.

        Args:
            code_string: The string containing Python code.

        Returns:
            True if all function calls are approved, False otherwise.
        """

        called_functions = self.parse_call(code_string)
        if not called_functions:  # Handle cases where parsing failed or no calls were found
            return True  # Consider it valid if no calls were found
        
        for func in called_functions:
            if func not in self.functions.keys():
                raise NotImplementedError(f"Do not use functions that are not provided!Forbidden function {func} found called in {code_string}.")
        
        return True
    
    def handle_function_call(self, call, context):

        try:
            self.validate_function_call(call)
            return safer_eval(call, context)
        except Exception as e:
            return e
    
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
            else:
                return [
                {
                    'call': None,
                    'result': f"Error parsing function list: Not list found."
                }
            ]
        
        # Get individual function calls
        try:
            calls = self.parse_function_calls(function_list_str)
        except Exception as e:
            return [
                {
                    'call': None,
                    'result': f"Error parsing function list: {e}"
                }
            ]

        context = self.global_context

        for call in calls:

            result = self.handle_function_call(call, context)

            results.append({
                'call': call,
                'result': result
            })

        return results
    