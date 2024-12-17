import argparse
from src.environment import *
from src.agent import *
from src.task import *
import os
import logging
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()

exploration_log = {}

def setup_logging(args):
    """Configure logging for the entire application"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{args.model.split('/')[-1]}_{args.explore_environment_iterations}_{args.benchmark_path}_{args.task}_{timestamp}.log"
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename)
        ]
    )

    logging.info(f"Experiment args: {args}")


def get_experiment_name(args):

    model = args.model.split("/")[-1]
    explore_model = args.explore_model.split("/")[-1] if args.explore_model else model
    bp = os.path.split(os.path.basename(args.benchmark_path))[-1]

    return f"{model}_temp_{args.execute_temp}_mipe_{args.max_iterations_per_episode}_{explore_model}_temp_{args.explore_temp}_eei_{args.explore_environment_iterations}_task_{bp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--model-type", "-mt", type=str, default="openai")
    parser.add_argument("--explore-model", "-em", type=str, default=None)
    parser.add_argument("--explore-model-type", "-emt", type=str, default="openai")
    parser.add_argument("--execute-temp", "-et", type=float, default=0.0)
    parser.add_argument("--explore-temp", "-ept", type=float, default=0.0)
    parser.add_argument("--explore-environment-iterations", "-eei", type=int, default=0)
    parser.add_argument("--max_iterations_per_episode", "-mipe", type=int, default=5)
    parser.add_argument("--benchmark_path", "-bp", type=str, default="nexus_relational_adversarial.py")
    parser.add_argument("--task", "-t", type=str, default="langchain_relational")

    args = parser.parse_args()

    # Set up logging first
    setup_logging(args)

    env = Environment(os.path.join("src", "benchmarks", args.benchmark_path)) # This file does not have return types or docstrings

    
    if args.explore_model:
        explore_agent = agent_registry[args.explore_model_type](args.explore_model, execute_temp=args.execute_temp, explore_temp=args.explore_temp)
    else:
        explore_agent = None
    
    agent = agent_registry[args.model_type](args.model, execute_temp=args.execute_temp, explore_temp=args.explore_temp)
    
    task = task_registry[args.task](env, agent, explore_agent=explore_agent, max_iterations_per_episode=args.max_iterations_per_episode, explore_environment_iterations=args.explore_environment_iterations)
    task.run_task()

    data = task.get_data()

    os.makedirs("experiments", exist_ok=True)

    with open(os.path.join('experiments', get_experiment_name(args)), 'w') as fout:
        json.dump(data, fout, indent=1)





if __name__ == "__main__":
    main()
