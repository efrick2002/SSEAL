import inspect
from typing import Callable
import re
import time
import os
import openai

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


def extract_function_calls(input_str):
    """
    Extracts function calls from a multi-line string and returns them as a list of strings.

    Args:
        input_str (str): A multi-line string containing function calls within square brackets.

    Returns:
        list: A list of function call strings.
    
    Example:
        input_str = "[
            get_current_user_id(),
            get_user_info(1),
            list_user_ids(),
            find_users_by_name("John"),
            get_user_location(1),
            get_location_info(1),
            get_user_favorite_foods(1),
            get_food_info(1)
        ]"
        
        result = extract_function_calls(input_str)
        print(result)
        
        # Output:
        # [
        #     "get_current_user_id()",
        #     "get_user_info(1)",
        #     "list_user_ids()",
        #     "find_users_by_name(\"John\")",
        #     "get_user_location(1)",
        #     "get_location_info(1)",
        #     "get_user_favorite_foods(1)",
        #     "get_food_info(1)"
        # ]
    """
    pattern = r'\b\w+\([^)]*\)'

    matches = re.findall(pattern, input_str)

    return matches

def safer_eval(input, context):
    try:
        return eval(input, context)
    except Exception as e:
        return e
    
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg,
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print(e)
            break
    return output


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):

    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60,
            )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(10)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except openai.APITimeoutError as e:
            print(type(e), "The api request timed out")
        except KeyError as e:
            print(type(e), e)
            break
        except Exception as e:
            print(type(e), e)
            break
    return output