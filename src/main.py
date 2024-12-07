import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--model-type", "-mt", type=str, default="openai")
    parser.add_argument("--explore-environment-iterations", "-eei", type=int, default=0)
    parser.add_argument("--max_iterations_per_episode", "-mipe", type=int, default=5)
    parser.add_argument("--benchmark_path", "-bp", type=str, default="nexus_relational_adversarial.py")
    args = parser.parse_args()

    # Set up logging first
    setup_logging()
    explore = True
    env = Environment(args.benchmark_path) # This file does not have return types or docstrings
    
    match args.model_type:
        case "openai":
            agent = OpenAIAgent(args.model, api_dict={"api_key": os.environ["OPENAI_API_KEY"], "api_base": None})
        
        case "anthropic":
            agent = AnthropicAgent(args.model, api_dict={"api_key": os.environ["ANTHROPIC_API_KEY"], "api_base": None})
        
        case _:
            raise NotImplemented("This type of agent is not yet implemented.")
    
    task = LangChainRelationalTask(env, agent, max_iterations_per_episode=args.max_iterations_per_episode, explore_environment_iterations=args.explore_environment_iterations)
    task.run_task()
    print('explore: ', explore)


if __name__ == "__main__":
    main()
