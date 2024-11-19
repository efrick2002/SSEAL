import inspect
from typing import Callable

def func_to_def(func: Callable) -> str:
    """
    Returns a string containing the function definition with its docstring.

    Args:
        func (Callable): The function to inspect.

    Returns:
        str: A string representation of the function's definition and docstring.
    """
    signature = inspect.signature(func)
    func_name = func.__name__
    
    return_annotation = signature.return_annotation
    if return_annotation is inspect.Signature.empty:
        return_type = ''
    else:
        return_type = f' -> {inspect.formatannotation(return_annotation)}'
    
    first_line = f'def {func_name}{signature}{return_type}:'
    
    doc = inspect.getdoc(func)
    if doc:
        doc_lines = doc.split('\n')
        indented_doc = '    """' + '\n    '.join(doc_lines) + '"""'
        func_def = f"{first_line}\n{indented_doc}"
    else:
        func_def = f"{first_line}\n    pass"
    
    return f"{func_def}\n"
