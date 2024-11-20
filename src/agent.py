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

class Agent(ABC):
    """Abstract base class for agents that can interact with environments."""
    
    API_MAX_RETRY = 16
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"

    def __init__(self):
        """Initialize the agent with empty state."""
        self.function_context = ""
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt: str = ""
    
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

        return f"""You are an AI agent that has can interact with an environment through function calls, which the environment will execute and return the results to you.

        Here is the list of functions you have access to:

        <functions>
        {self.function_context}
        </functions>
        
        To execute a function, you should output a series of python function calls, seperated by newlines, all in between <function_list> and </function_list> tags. 
        
        <function_list>
        f(1)
        g()
        h(4, 'a')
        </function_list>

        You will recieve the outputs from the function calls as a list of dictionaries of the function calls and their corresponding results in between <function_results> and </function_results> tags.
        """

    def update_function_context(self, context: str) -> None:
        """Update the function context with new information."""
        self.function_context = context

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
    def generate_response(self, temperature: float = 0.0, max_tokens: int = 8192) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: Generated response
        """
        pass
    
    def explore_environment(self, iterations: int = 2):
        """Procedure to explore the environment, should ultimately use the update_function_context() 
        function to modify the signatures with updated and more reliable information""" 
        with open("prompts/prompt2.txt") as fin:
            prompt = fin.read()
        self.function_call_log = []
        prompt = prompt.replace("{{FUNCTIONS}}", self.function_context)
        logging.info('Exploration prompt\n' + prompt)

        for i in range(iterations):
            completion = self.generate_response(prompt)
            logging.info('completion: ' + completion)
            env_response = self.environment.execute_function_list(completion)
            self.function_call_log += env_response
            prompt = f'Output from the environment, formatted as a list of dictionaries describing the function called, and its output: {env_response}\n'
            if i < iterations - 1:
                prompt += f'\nCarefully analyze this output, and propose additional function calls based on which functions you believe need more clarification. Use the same structure as before\n'
 
        with open("prompts/prompt3.txt") as fin:
            final_prompt = fin.read()

        final_prompt = prompt + '\n' + final_prompt.replace("{{FUNCTIONS}}", self.function_context)
        logging.info('final prompt: \n' + final_prompt)
        completion = self.generate_response(final_prompt)
        logging.info('final completion: \n' + completion)
        new_context_pattern = r'<new context>(.*?)</new context>'
        matches = re.findall(new_context_pattern, completion, re.DOTALL)
        if matches:
            new_function_context = matches[0]
            self.update_function_context(new_function_context)
            logging.info('Updated function context')

class OpenAIAgent(Agent):
    """Implementation of an agent that uses OpenAI's API."""

    
    def __init__(self, model: str = "gpt-4", api_dict: Optional[Dict[str, str]] = None):
        """
        Initialize the OpenAI agent.
        
        Args:
            model (str): OpenAI model to use
            api_dict (Optional[Dict[str, str]]): API configuration dictionary
        """
        super().__init__()
        self.model = model
        self.api_dict = api_dict
        
        if api_dict:
            self.client = openai.OpenAI(
                base_url=api_dict["api_base"],
                api_key=api_dict["api_key"],
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
                    max_tokens=max_tokens,
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
                break
                
        return output

class AnthropicAgent(Agent):
    """Implementation of an agent that uses Anthropic's API."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_dict: Optional[Dict[str, str]] = None):
        """
        Initialize the Anthropic agent.
        
        Args:
            model (str): Anthropic model to use
            api_dict (Optional[Dict[str, str]]): API configuration dictionary
        """
        super().__init__()
        self.model = model
        self.api_dict = api_dict
        
        if api_dict:
            self.api_key = api_dict["api_key"]
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
                break
                    
        return output