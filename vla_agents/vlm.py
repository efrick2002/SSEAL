import logging
import os
import torch
import numpy as np
import google.generativeai as genai
import cv2
from PIL import Image

class VLM:
    """
    Base class for a Vision-Language Model (VLM) agent. 
    This class should be extended to implement specific VLMs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VLM agent with optional parameters.
        """
        self.name = "not implemented"

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference with the VLM agent, passing images and a text prompt.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError
    
    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the VLM, incorporating past context.

        Parameters
        ----------
        history : int
            The number of context steps to keep for inference.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def rewind(self):
        """
        Rewind the VLM agent one step by removing the last inference context.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total cost or spend associated with the agent.
        """
        return 0


class GeminiVLM(VLM):
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """

    def __init__(self, model="gemini-1.5-flash", system_instruction=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        # Configure generation parameters
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 500,
            "response_mime_type": "text/plain",
        }

        self.spend = 0
        self.cost_per_input_token = 0.075 / 1_000_000 if 'flash' in self.name else 1.25 / 1_000_000
        self.cost_per_output_token = 0.3 / 1_000_000 if 'flash' in self.name else 5 / 1_000_000
        
        # Initialize Gemini model and chat session
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
            system_instruction=system_instruction
        )
        self.session = self.model.start_chat(history=[])

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the Gemini model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.session.send_message([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

            # Manage history length based on the number of past steps to keep
            if history == 0:
                self.session = self.model.start_chat(history=[])
            elif len(self.session.history) > 2 * history:
                self.session.history = self.session.history[-2 * history:]
        
        except Exception as e:  
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text
    
    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.session.history) > 1:
            self.model.rewind()

    def reset(self):
        """
        Reset the chat history.
        """
        self.session = self.model.start_chat(history=[])

    def call_video(self, video_file, text_prompt):
        try:
            response = self.session.send_message([video_file, text_prompt])
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

        except Exception as e:  
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.model.generate_content([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

        except Exception as e:  
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text
    
    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend
