from environment import *
from agent import *
from task import *
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

def setup_logging():
    """Configure logging for the entire application"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/app_{timestamp}.log"
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename)
        ]
    )

def main():
    """Runs the LangChainRelational task with a default agent using the function signitures defined in nexus_relational.py"""
    # Set up logging first
    setup_logging()
    explore = False
    env = Environment("nexus_relational_nosig.py") # This file does not have return types or docstrings
    agent = OpenAIAgent("gpt-4o-mini", api_dict={"api_key": os.environ["OPENAI_API_KEY"], "api_base": None})
    task = LangChainRelationalTask(env, agent, max_iterations_per_episode=5, explore_environment=explore)
    task.run_task()
    print('explore: ', explore)


if __name__ == "__main__":
    main()
