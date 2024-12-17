from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import json
import time
import os
import openai
import anthropic
from anthropic import HUMAN_PROMPT
import logging
import re
from pprint import pprint
import json
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

class Agent(ABC):
    """Abstract base class for agents that can interact with environments."""
    
    API_MAX_RETRY = 16
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"

    def __init__(self, model: str = "gpt-4o", explore_temp=0.0, execute_temp=0.0):
        """Initialize the agent with empty state."""
        self.function_context = ""
        self.additional_information = ""
        self.examples = ""
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt: str = ""

        self.model: str = model

        self.explore_temp: float = explore_temp
        self.execute_temp: float = execute_temp

        self.client = None


    
    def register_environment(self, environment) -> None:
        """
        Register an environment with the agent, loading available functions.
        
        Args:
            environment: An instance of Environment class
        """
        self.environment = environment
        self.function_context = environment.get_function_context()
    
    def make_function_context_prompt(self) -> str:
        """
        Generate a prompt for the function context.
        
        Returns:
            str: Generated prompt
        """

        return f"""You are an AI agent that can interact with an environment through function calls, in which the environment will execute and return the results to you.

        You will recieve a query from the user. Find the function call that satisfies the query. Use the environments output to verify the function call was correct.

        Here is the list of functions you have access to:

        <functions>
        {self.function_context}
        </functions>

        Here is additional information regarding the evironment and how to interact and use functions, use this to guide your query answering process:

        <additional_information>
        {self.additional_information}
        </additional_information>

        Here are some additional examples:

        <examples>
        {self.examples}
        </examples>
        
        Once you recieve the query, think step by step to determine which function(s) to call. Output the function(s) you would like to call seperated by newlines, all in between the <function_list> and </function_list> tags. 
        
        Like this example:

        <function_list>
        f(1)
        g()
        h(4, 'a')
        </function_list>

        After providing a list of functions to call, the environment will execute these functions and you will recieve the outputs from the function calls as a list of dictionaries of the function calls and their corresponding results.

        """

    def update_function_context(self, context: str) -> None:
        """Update the function context with new information."""
        self.function_context = context
    
    def update_additional_information(self, context: str) -> None:
        """Update the additional information with new information."""
        self.additional_information = context
    
    def update_examples(self, context: str) -> None:
        """Update the examples with new information."""
        self.examples = context

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt for the agent.
        
        Args:
            prompt (str): System prompt to set
        """
        self.system_prompt = prompt
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = prompt
        else:
            self.conversation_history.insert(0, {"role": "system", "content": prompt})
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            message (str): User message to add
        """
        self.conversation_history.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            message (str): Assistant message to add
        """
        self.conversation_history.append({"role": "assistant", "content": message})
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history, preserving the system prompt if set."""
        if self.system_prompt:
            self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.conversation_history = []
    
    @abstractmethod
    def generate_response(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: Generated response
        """
        pass
    
    def explore_environment(self, iterations: int = 8):
        """Procedure to explore the environment, should ultimately use the update_function_context() 
        function to modify the signatures with updated and more reliable information""" 

        env_name = os.path.splitext(os.path.basename(self.environment.functions_file_path))[0]


        os.makedirs(f"caches/{env_name}", exist_ok=True)

        model_name = self.model.split("/")[-1]

        cache_path = f"caches/{env_name}/{model_name}.json"
        
        if os.path.exists(cache_path):

            with open(cache_path, 'r') as fin:

                cache = json.load(fin)
        
        else:
            cache = {}
        

        if str(iterations) in cache:

            logging.info('Exploration results loaded from cache.')

            new_function_context = cache[str(iterations)]['new_function_context']
            additional_info = cache[str(iterations)].get('additional_information', '')
            examples = cache[str(iterations)].get('examples', '')

            self.update_function_context(new_function_context)
            self.update_additional_information(additional_info)
            self.update_examples(examples)

            return

        cache[str(iterations)] = {}

        with open("prompts/prompt2.txt") as fin:
            prompt = fin.read()
        self.function_call_log = []
        prompt = prompt.replace("{{FUNCTIONS}}", self.function_context)
        logging.info('Exploration prompt\n' + prompt)

        cache[str(iterations)]['prompt'] = prompt
        
        completions = []
        env_responses = []

        for i in range(iterations):
            completion = self.generate_response(prompt, temperature=self.explore_temp)
            completions.append(completion)
            
            logging.info('completion: ' + completion)
            env_response = self.environment.execute_function_list(completion)
            env_responses.append(str(env_response))
            logging.info(f"Call Outputs: {env_response}")
            self.function_call_log += env_response
            prompt = f'Output from the environment, formatted as a list of dictionaries describing the function called, and its output: {env_response}\n'
            if i < iterations - 1:
                prompt += f'\nCarefully analyze this output, and propose additional function calls based on which functions you believe need more clarification. Use the same structure as before.\n'
 
        cache[str(iterations)]['completions'] = completions
        cache[str(iterations)]['env_responses'] = env_responses

        with open("prompts/prompt3.txt") as fin:
            final_prompt = fin.read()

        final_prompt = prompt + '\n' + final_prompt.replace("{{FUNCTIONS}}", self.function_context)
        cache[str(iterations)]['final_prompt'] = final_prompt
        logging.info('final prompt: \n' + final_prompt)
        completion = self.generate_response(final_prompt, temperature=self.execute_temp)
        logging.info('final completion: \n' + completion)

        cache[str(iterations)]['final_completion'] = completion

        new_context_pattern = r'<functions>(.*?)</functions>'
        matches = re.findall(new_context_pattern, completion, re.DOTALL)
        if matches:
            new_function_context = matches[0]
            self.update_function_context(new_function_context)
            logging.info('Updated function context')

            cache[str(iterations)]['new_function_context'] = new_function_context
        
        new_context_pattern = r'<additional_information>(.*?)</additional_information>'
        matches = re.findall(new_context_pattern, completion, re.DOTALL)
        if matches:
            additional_info = matches[0]
            self.update_additional_information(additional_info)
            logging.info('Updated additional information')

            cache[str(iterations)]['additional_information'] = additional_info

        new_context_pattern = r'<examples>(.*?)</examples>'
        matches = re.findall(new_context_pattern, completion, re.DOTALL)
        if matches:
            examples = matches[0]
            self.update_examples(examples)
            logging.info('Updated examples')

            cache[str(iterations)]['examples'] = examples


        cache[str(iterations)]['conversation_history'] = self.conversation_history

        with open(cache_path, 'w') as fout:

            json.dump(cache, fout, indent=1)
    
    def transfer_knowledge_from(self, other_agent):
        self.function_context = other_agent.function_context
        self.additional_information = other_agent.additional_information
        self.examples = other_agent.examples

agent_registry: Dict[str, Agent] = {}

def register_agent(name: str):
    def decorator(cls):
        agent_registry[name] = cls
        return cls
    return decorator

@register_agent("openai")
class OpenAIAgent(Agent):
    """Implementation of an agent that uses OpenAI's API."""

    api_dict: Optional[Dict[str, str]] = {"api_key": os.environ.get("OPENAI_API_KEY", "-"), "api_base": None}
    
    def __init__(self, model: str = "gpt-4o", explore_temp=0.0, execute_temp=0.0):
        """
        Initialize the OpenAI agent.
        
        Args:
            model (str): OpenAI model to use
            api_dict (Optional[Dict[str, str]]): API configuration dictionary
        """
        super().__init__(model=model, explore_temp=explore_temp, execute_temp=execute_temp)
        
        if self.api_dict:
            self.client = openai.OpenAI(
                base_url=self.api_dict["api_base"],
                api_key=self.api_dict["api_key"],
            )
        else:
            self.client = openai.OpenAI()
    
    def generate_response(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192, add_to_history: bool = True) -> str:
        """
        Generate a response using OpenAI's API.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens in response
            add_to_history (bool): Whether to add the response to the conversation history
            
        Returns:
            str: Generated response
        """
        output = self.API_ERROR_OUTPUT
        self.conversation_history.append({"role": "user", "content": prompt})
        
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=temperature,
                    max_tokens=None,
                    timeout=60,
                )
                output = completion.choices[0].message.content
                if add_to_history:
                    self.add_assistant_message(output)
                break
            except openai.RateLimitError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            except openai.BadRequestError as e:
                print(self.conversation_history)
                print(type(e), e)
                break
            except openai.APITimeoutError as e:
                print(type(e), "The api request timed out")
                time.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
                
        return output.replace("```python", "<function_list>").replace("```", "</function_list>")

@register_agent("gemini")
class GeminiAgent(OpenAIAgent):

    api_dict: Optional[Dict[str, str]] = {"api_key": os.environ.get("GEMINI_API_KEY", "-"), "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/"}

@register_agent("qwq")
class QwQAgent(OpenAIAgent):

    api_dict: Optional[Dict[str, str]] = {"api_key": os.environ.get("QWEN_API_KEY", "-"), "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"}

@register_agent("fireworks")
class FireworkAgent(OpenAIAgent):

    api_dict: Optional[Dict[str, str]] = {"api_key": os.environ.get("FIREWORKS_API_KEY", "-"), "api_base": "https://api.fireworks.ai/inference/v1"}

@register_agent("anthropic")
class AnthropicAgent(Agent):
    """Implementation of an agent that uses Anthropic's API."""

    api_dict: Optional[Dict[str, str]] = {"api_key": os.environ.get("ANTHROPIC_API_KEY", "-"), "api_base": None}
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", explore_temp=0.0, execute_temp=0.0):
        """
        Initialize the Anthropic agent.
        
        Args:
            model (str): Anthropic model to use
            api_dict (Optional[Dict[str, str]]): API configuration dictionary
        """
        super().__init__(model=model, explore_temp=explore_temp, execute_temp=execute_temp)
        
        if self.api_dict:
            self.api_key = self.api_dict["api_key"]
        else:
            self.api_key = os.environ["ANTHROPIC_API_KEY"]
            
        self.client = anthropic.Anthropic(api_key=self.api_key)

    
    def generate_response(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192, add_to_history: bool = True) -> str:
        """
        Generate a response using Anthropic's API.
        
        Args:
            prompt (str): Prompt for the LLM
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens in response
            add_to_history (bool): Whether to add the response to the conversation history
            
        Returns:
            str: Generated response
        """
        output = self.API_ERROR_OUTPUT
        
        # Extract system message if present
        messages = self.conversation_history
        sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        messages = messages[1:] if sys_msg else messages
        messages.append({"role": "user", "content": prompt})
        
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    stop_sequences=[HUMAN_PROMPT],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=sys_msg,
                )
                output = response.content[0].text
                if add_to_history:
                    self.add_assistant_message(output)
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
                    
        return output