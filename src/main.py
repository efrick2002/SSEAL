from environment import *
from agent import *
from task import *
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    """Runs the LangChainRelational task with a default agent using the function signitures defined in nexus_relational.py"""
    env = Environment("nexus_relational.py")
    agent = OpenAIAgent("gpt-4o-mini", api_dict={"api_key": os.environ["OPENAI_API_KEY"], "api_base": None})
    task = LangChainRelationalTask("LangChainRelational", env, agent, max_iterations_per_episode=3)
    task.run_task()


if __name__ == "__main__":
    main()
