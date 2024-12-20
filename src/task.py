import pandas as pd
import logging
import os
import re
from datetime import datetime
from src.environment import Environment
from src.agent import Agent
from copy import deepcopy

class Task:
    name = "Base Task"
    df = None

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        explore_agent: Agent = None,
        max_iterations_per_episode=3,
        explore_environment_iterations=0,
    ):
        self.environment = environment
        self.agent = agent
        self.episodes_completed = 0
        self.successful_episodes = 0
        self.num_episodes = 0
        self.total_iterations = 0
        logging.info(f"Initialized task: {self.name}")

        self.num_episodes = len(self.df)
        self.agent.register_environment(environment)

        self.data = []

        if explore_environment_iterations:

            explore_agent = explore_agent if explore_agent is not None else self.agent

            explore_agent.register_environment(environment)

            explore_agent.explore_environment(iterations=explore_environment_iterations)

            self.agent.transfer_knowledge_from(explore_agent)

            self.environment.reset_original_context()

        self.agent.set_system_prompt(self.agent.make_function_context_prompt())

        logging.info("Execution Agent System Prompt:\n\n" + self.agent.system_prompt + "\n\n")

        if max_iterations_per_episode == 0:
            logging.info(self.agent.system_prompt)
            exit(0)

        self.max_iterations = max_iterations_per_episode

    def run_task(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.num_episodes
        for i in range(num_episodes):
            self.run_episode(i)
        print(self.get_metrics())
        logging.info(f"\n\nFinal Scores:\n{self.get_metrics()}")

    def get_data(self):
        return self.data
    
    def get_query(self, data_ndx):
        raise NotImplementedError()
    
    def get_ground_truth(self, data_ndx):
        raise NotImplementedError()

    def run_episode(self, data_ndx):

        self.agent.clear_conversation_history()
        user_query = self.get_query(data_ndx)
        ground_truth = self.get_ground_truth(data_ndx)

        logging.info(f"\n{'='*20} Starting episode {data_ndx} {'='*20}")
        logging.info(f"User query: {user_query}")
        logging.info(f"Ground truth answers: {ground_truth}")

        prompt = f"Query: {user_query}\n"

        env_response = [{"call": None, "result": None}]

        env_responses = []
        completions = []

        for i in range(self.max_iterations):
            self.total_iterations += 1
            logging.info(f"Iteration {i} - Prompt: {prompt}")
            completion = self.agent.generate_response(
                prompt, temperature=self.agent.execute_temp
            )

            completions.append(completion)

            logging.info(f"Model response: {completion}")

            if (
                "<end_query>" in completion
                or i == self.max_iterations - 1
                or "<function_list>" not in completion
            ):

                if "<function_list>" in completion:
                    env_response = self.environment.execute_function_list(completion)
                    env_responses.append(str(env_response))

                initial_successful = self.successful_episodes
                self.evaluate_answer(env_response, ground_truth)
                success = self.successful_episodes > initial_successful
                logging.info(
                    f"Episode {data_ndx} completed - Answer: {env_response} - Success: {success}"
                )
                break
            else:
                env_response = self.environment.execute_function_list(completion)
                env_responses.append(str(env_response))
                prompt = f"Output from the environment: {env_response}\n If you believe this output satisfies the user's query, output `<end_query>`, otherwise continue calling more functions."
        
        data = {
            'idx': data_ndx,
            'query': user_query,
            'completions': completions,
            'env_responses': env_responses,
            'num_iter': i + 1,
            'conversation_history': self.agent.conversation_history,
            'success': success,

        }

        self.data.append(data)

        print("\n\n" + "=" * 50 + "\n\n", file=open("debug.txt", "a"))
        print(self.agent.conversation_history, file=open("debug.txt", "a"))
        print("\n\n" + "=" * 50 + "\n\n", file=open("debug.txt", "a"))

        self.episodes_completed += 1
        if i == self.max_iterations - 1:
            logging.warning(
                f"Episode {data_ndx} reached max iterations without finding an answer"
            )

    def get_metrics(self):
        return {
            "name": self.name,
            "episodes_completed": self.episodes_completed,
            "successful_episodes": self.successful_episodes,
            "success_rate": (
                self.successful_episodes / self.episodes_completed
                if self.episodes_completed > 0
                else 0
            ),
            "itr/episode": (
                self.total_iterations / self.episodes_completed
                if self.episodes_completed > 0
                else 0
            ),
        }


task_registry: dict[str, Task] = {}


def register_task(name: str):
    def decorator(cls):
        task_registry[name] = cls
        return cls

    return decorator


@register_task("langchain_relational")
class LangChainRelationalTask(Task):
    name = "LangChainRelationalTask"
    df = pd.read_parquet(
        "hf://datasets/Nexusflow/LangChainRelational/data/train-00000-of-00001.parquet"
    )

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        explore_agent: Agent = None,
        max_iterations_per_episode=3,
        explore_environment_iterations=0,
    ):
        super().__init__(
            environment,
            agent,
            explore_agent=explore_agent,
            max_iterations_per_episode=max_iterations_per_episode,
            explore_environment_iterations=explore_environment_iterations,
        )

    # def _task_prompt(self):
    #     return """Your task is to answer a user query using the available functions in the environment. Always output your thinking and reasoning. If you think you know the answer to the question, return your answer in between <answer> and </answer> tags. """

    def evaluate_answer(self, env_response, gt_answers):

        agent_answer = [er['result'] for er in env_response]

        agent_answer = list(map(str, agent_answer))
        for gt_answer in gt_answers:
            if str(gt_answer) in agent_answer:
                self.successful_episodes += 1 / len(gt_answers)


            # print("agent: ", agent_answer)
            # print("gt: ", gt_answer)
            # print("t: ", str(gt_answer) in agent_answer)
        self.environment.reset_original_context()


    def get_query(self, data_ndx):
        return self.df.iloc[data_ndx]["user_query"]

    def get_ground_truth(self, data_ndx):

        ground_truth = self.df.iloc[data_ndx]["ground_truth"]
        
        return ground_truth
    
    # def run_episode(self, data_ndx):

    #     self.agent.clear_conversation_history()
    #     user_query = self.get_query(data_ndx)
    #     ground_truth = self.get_ground_truth(data_ndx)

    #     logging.info(f"\n{'='*20} Starting episode {data_ndx} {'='*20}")
    #     logging.info(f"User query: {user_query}")
    #     logging.info(f"Ground truth answers: {ground_truth}")

    #     prompt = f"Query: {user_query}\n"

    #     env_response = [{"call": None, "result": None}]

    #     for i in range(self.max_iterations):
    #         self.total_iterations += 1
    #         logging.info(f"Iteration {i} - Prompt: {prompt}")
    #         completion = self.agent.generate_response(
    #             prompt, temperature=self.agent.execute_temp
    #         )
    #         logging.info(f"Model response: {completion}")

    #         if (
    #             "<end_query>" in completion
    #             or i == self.max_iterations - 1
    #             or "<function_list>" not in completion
    #         ):

    #             if "<function_list>" in completion:
    #                 env_response = self.environment.execute_function_list(completion)

    #             initial_successful = self.successful_episodes
    #             self.evaluate_answer([er['result'] for er in env_response], ground_truth)
    #             success = self.successful_episodes > initial_successful
    #             logging.info(
    #                 f"Episode {data_ndx} completed - Answer: {[er['result'] for er in env_response]} - Success: {success}"
    #             )
    #             break
    #         else:
    #             env_response = self.environment.execute_function_list(completion)
    #             prompt = f"Output from the environment: {env_response}\n If you believe this output satisfies the user's query, output `<end_query>`, otherwise continue calling more functions."
        
    #     print("\n\n" + "=" * 50 + "\n\n", file=open("debug.txt", "a"))
    #     print(self.agent.conversation_history, file=open("debug.txt", "a"))
    #     print("\n\n" + "=" * 50 + "\n\n", file=open("debug.txt", "a"))

    #     self.episodes_completed += 1
    #     if i == self.max_iterations - 1:
    #         logging.warning(
    #             f"Episode {data_ndx} reached max iterations without finding an answer"
    #         )
    # def run_episode(self, data_ndx):

    #     self.agent.clear_conversation_history()
    #     user_query = self.df.iloc[data_ndx]["user_query"]
    #     ground_truth = self.df.iloc[data_ndx]["ground_truth"]
    #     gt_answers = set()
    #     for ans in ground_truth:
    #         gt_answers.add(str(ans))

    #     logging.info(f"\n{'='*20} Starting episode {data_ndx} {'='*20}")
    #     logging.info(f"User query: {user_query}")
    #     logging.info(f"Ground truth answers: {gt_answers}")

    #     prompt = f"{self._task_prompt()}\n User query: {user_query}\n"
    #     for i in range(self.max_iterations):
    #         self.total_iterations += 1
    #         logging.info(f"Iteration {i} - Prompt: {prompt}")
    #         completion = self.agent.generate_response(
    #             prompt, temperature=self.agent.execute_temp
    #         )
    #         logging.info(f"Model response: {completion}")

    #         if "<answer>" in completion:
    #             answer_match = re.search(
    #                 r"<answer>(.*?)</answer>", completion, re.DOTALL
    #             )
    #             if answer_match:
    #                 agent_answer = answer_match.group(1)
    #                 initial_successful = self.successful_episodes
    #                 self.evaluate_answer(agent_answer, gt_answers)
    #                 success = self.successful_episodes > initial_successful
    #                 logging.info(
    #                     f"Episode {data_ndx} completed - Answer: {agent_answer} - Success: {success}"
    #                 )
    #                 break
    #             else:
    #                 logging.warning(
    #                     "Found <answer> tag but couldn't extract answer text"
    #                 )
    #         else:
    #             env_response = self.environment.execute_function_list(completion)
    #             prompt = f"Output from the environment: {env_response}\n"

    #     self.episodes_completed += 1
    #     if i == self.max_iterations - 1:
    #         logging.warning(
    #             f"Episode {data_ndx} reached max iterations without finding an answer"
    #         )


@register_task("sports_data")
class SportsDataTask(Task):
    from src.benchmarks.sports import QUESTIONS

    name = "SportsData"
    df = list(QUESTIONS.items())

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        explore_agent: Agent = None,
        max_iterations_per_episode=3,
        explore_environment_iterations=0,
    ):
        super().__init__(
            environment,
            agent,
            explore_agent=explore_agent,
            max_iterations_per_episode=max_iterations_per_episode,
            explore_environment_iterations=explore_environment_iterations,
        )

    # def _task_prompt(self):
    #     return """Your task is to satisfy a user query using the available functions in the environment. Always output your thinking and reasoning before making a function call(s). If you believe your previous function call has satisfied the user's query upon observing the output from the function call, output `<end_query>` for the current turn and the user will recieve the output of the last function call.
        
    #     Remember, you are interacting with a function environment, not a user. The user will only see the output of your last function call."""
    def get_query(self, data_ndx):
        return self.df[data_ndx][0]

    def get_ground_truth(self, data_ndx):
        return self.df[data_ndx][1]

    def evaluate_answer(self, env_response, gt_df: pd.DataFrame):

        agent_df = env_response[-1]["result"]

        if type(gt_df) == str:
            self.successful_episodes += int(gt_df == str(agent_df))
        else:
            self.successful_episodes += int(gt_df.equals(agent_df))

        self.environment.reset_original_context()

    # def run_episode(self, data_ndx):

    #     self.agent.clear_conversation_history()
    #     user_query = self.df[data_ndx][0]
    #     ground_truth = self.df[data_ndx][1]

    #     logging.info(f"\n{'='*20} Starting episode {data_ndx} {'='*20}")
    #     logging.info(f"User query: {user_query}")
    #     logging.info(f"Ground truth answers: {ground_truth}")

    #     prompt = f"Query: {user_query}\n"

    #     env_response = [{"call": None, "result": None}]

    #     for i in range(self.max_iterations):
    #         self.total_iterations += 1
    #         logging.info(f"Iteration {i} - Prompt: {prompt}")
    #         completion = self.agent.generate_response(
    #             prompt, temperature=self.agent.execute_temp
    #         )
    #         logging.info(f"Model response: {completion}")

    #         if (
    #             "<end_query>" in completion
    #             or i == self.max_iterations - 1
    #             or "<function_list>" not in completion
    #         ):

    #             if "<function_list>" in completion:
    #                 env_response = self.environment.execute_function_list(completion)

    #             initial_successful = self.successful_episodes
    #             self.evaluate_answer(env_response[-1]["result"], ground_truth)
    #             success = self.successful_episodes > initial_successful
    #             logging.info(
    #                 f"Episode {data_ndx} completed - Answer: {env_response[-1]['result']} - Success: {success}"
    #             )
    #             break
    #         else:
    #             env_response = self.environment.execute_function_list(completion)
    #             prompt = f"Output from the environment: {env_response}\n If you believe this output satisfies the user's query, output `<end_query>`, otherwise continue calling more functions."
        
    #     print("\n\n" + "=" * 50 + "\n\n", file=open("debug.txt", "a"))
    #     print(self.agent.conversation_history, file=open("debug.txt", "a"))
    #     print("\n\n" + "=" * 50 + "\n\n", file=open("debug.txt", "a"))

    #     self.episodes_completed += 1
    #     if i == self.max_iterations - 1:
    #         logging.warning(
    #             f"Episode {data_ndx} reached max iterations without finding an answer"
    #         )

@register_task("linux_terminal")
class LinuxTerminalTask(Task):
    from src.benchmarks.linux_terminal import QUESTIONS, _run_solution

    name = "LinuxTerminal"
    df = QUESTIONS
    run_solution = _run_solution

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        explore_agent: Agent = None,
        max_iterations_per_episode=3,
        explore_environment_iterations=0,
    ):
        super().__init__(
            environment,
            agent,
            explore_agent=explore_agent,
            max_iterations_per_episode=max_iterations_per_episode,
            explore_environment_iterations=explore_environment_iterations,
        )

    def get_query(self, data_ndx):
        global_context = self.environment.global_context

        code = f"_set_state({deepcopy(self.df[data_ndx]['initial_state'])})"

        eval(code, global_context)

        return self.df[data_ndx]['query']

    def get_ground_truth(self, data_ndx):

        init_state = deepcopy(self.df[data_ndx]['initial_state'])
        gt_solution = self.df[data_ndx]['ground_truth_solution_steps']

        return LinuxTerminalTask.run_solution(init_state, gt_solution)

    def evaluate_answer(self, agent_output, gt_env):

        agent_env = self.environment.global_context['_env']
        
        self.successful_episodes += int(gt_env == agent_env)

        # print("correct: ", gt_env == agent_env, gt_env.fs == agent_env.fs, agent_env.env == gt_env.env)
        # print("agent env: ", agent_env.fs, agent_env.env)
        # print("gt env: ", gt_env.fs, gt_env.env)
        logging.info(f"Observed Agent State: {agent_env.fs} {agent_env.env}")
        logging.info(f"Expected Agent State: {gt_env.fs} {gt_env.env}")


        
        self.environment.reset_original_context()


@register_task("custom")
class Custom(Task):

    name = "Custom"
    df = []

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        explore_agent: Agent = None,
        max_iterations_per_episode=3,
        explore_environment_iterations=0,
    ):
        super().__init__(
            environment,
            agent,
            explore_agent=explore_agent,
            max_iterations_per_episode=max_iterations_per_episode,
            explore_environment_iterations=explore_environment_iterations,
        )