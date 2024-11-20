import pandas as pd
import logging
import os
import re
from datetime import datetime
from environment import Environment
from agent import Agent

class Task:
    name = 'Base Task'

    def __init__(self, environment: Environment, agent: Agent):
        self.environment = environment
        self.agent = agent
        self.episodes_completed = 0
        self.successful_episodes = 0
        self.num_episodes = 0
        self.total_iterations = 0
        logging.info(f"Initialized task: {self.name}")

    def run_task(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.num_episodes
        for i in range(num_episodes):
            self.run_episode(i)
        print(self.get_metrics())

    def run_episode(self, data_ndx):
        raise NotImplementedError
    
    def get_metrics(self):
        return {
            "name": self.name,
            "episodes_completed": self.episodes_completed,
            "successful_episodes": self.successful_episodes,
            "success_rate": self.successful_episodes / self.episodes_completed if self.episodes_completed > 0 else 0,
            'itr/episode': self.total_iterations / self.episodes_completed if self.episodes_completed > 0 else 0
        }


class LangChainRelationalTask(Task):
    name = 'LangChainRelationalTask'

    def __init__(self, environment: Environment, agent: Agent, max_iterations_per_episode=3, explore_environment=False):

        super().__init__(environment, agent)
        self.df = pd.read_parquet("hf://datasets/Nexusflow/LangChainRelational/data/train-00000-of-00001.parquet")
        self.num_episodes = len(self.df)    
        self.agent.register_environment(environment)
        if explore_environment:
            self.agent.explore_environment()
        self.agent.set_system_prompt(self.agent.make_function_context_prompt())
        self.max_iterations = max_iterations_per_episode

    def _task_prompt(self):
        return """Your task is to answer a user query using the available functions in the environment. Always output your thinking and reasoning. If you think you know the answer to the question, return your answer in between <answer> and </answer> tags. """ 

    def evaluate_answer(self, agent_answer, gt_answers):
        for gt_answer in gt_answers:
            if gt_answer in agent_answer:
                self.successful_episodes += 1/len(gt_answers)

    def run_episode(self, data_ndx):

        self.agent.clear_conversation_history()
        user_query = self.df.iloc[data_ndx]["user_query"]
        ground_truth = self.df.iloc[data_ndx]["ground_truth"]
        gt_answers = set()
        for ans in ground_truth:
            gt_answers.add(str(ans))

        logging.info(f"\n{'='*20} Starting episode {data_ndx} {'='*20}")
        logging.info(f"User query: {user_query}")
        logging.info(f"Ground truth answers: {gt_answers}")
        
        prompt = f'{self._task_prompt()}\n User query: {user_query}\n'
        for i in range(self.max_iterations):
            self.total_iterations += 1
            logging.info(f"Iteration {i} - Prompt: {prompt}")
            completion = self.agent.generate_response(prompt)
            logging.info(f"Model response: {completion}")
            
            if "<answer>" in completion:
                answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
                if answer_match:
                    agent_answer = answer_match.group(1)
                    initial_successful = self.successful_episodes
                    self.evaluate_answer(agent_answer, gt_answers)
                    success = self.successful_episodes > initial_successful
                    logging.info(f"Episode {data_ndx} completed - Answer: {agent_answer} - Success: {success}")
                    break
                else:
                    logging.warning("Found <answer> tag but couldn't extract answer text")
            else:
                env_response = self.environment.execute_function_list(completion)
                prompt = f'Output from the environment: {env_response}\n'

        self.episodes_completed += 1
        if i == self.max_iterations - 1:
            logging.warning(f"Episode {data_ndx} reached max iterations without finding an answer")
