from vlm import GeminiVLM
import json
from dotenv import load_dotenv
import ast

load_dotenv()
agent = GeminiVLM(model="gemini-1.5-pro", system_instruction="You are an language assistant that helps generate data by rephrasing robotics tasks")

def make_icl_prompt(instruction):

    return f"""
    Your task is to rephrase a specific robotic instruction. Here are some examples:

    instruction: close the top drawer of the cabinet
    rephrased: I need all the drawers of the cabinet to be closed, especially the top one
    instruction: put the chocolate pudding to the right of the plate
    rephrased: find the desert item. I want it to be on the right side of the plate
    instruction: pick up the black bowl on the ramekin and place it on the plate
    rephrased: Whatever is on the ramekin right now, it should be moved onto the plate

    Please generate 2 different rephrases for the following instruction, be creative:
    {instruction}
    Return them in the json {{1: <rephrase1>, 2: <rephrase2>}}
    """

    
data = {}

tasks = json.load(open("tasks.json"))
for task, lst in tasks.items():
    data[task] = {}
    for instruction in lst[0:30]:
        try:
            lst = []
            result = agent.call(images=[], text_prompt=make_icl_prompt(instruction))
            d = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1])
            for k, v in d.items():
                lst.append(v)
            data[task][instruction] = lst
            # print(result)
        except Exception as e:
            print(f"Error: {e}")

with open("rephrased.json", "w") as f:
    json.dump(data, f)