import json
import random
from dotenv import load_dotenv

load_dotenv()

def make_icl_instructions(suite, instruction, examples=20, plain=False):
    pos = []
    neg = []
    for task in ['goal', 'object', 'spatial']:
        if task != suite:
            try:
                results = json.load(open(f"json_stuff/libero_{task}_reprhase2_.json")) + json.load(open(f"json_stuff/libero_{task}_reprhase2.json"))
                for i in range(len(results)):
                    if results[i]['success']:
                        pos.append([results[i]['rephrased_instruction'], True])
                    else:
                        neg.append([results[i]['rephrased_instruction'], False])
            except Exception as e:
                print(e)
                continue
    print(len(pos), len(neg ))
    # results = json.load(open('exploration_logs.json'))
    # for result in results:
    #     if result['outcome']:
    #         pos.append([result['instruction'], True])
    #     else:
    #         neg.append([result['instruction'], False])
    import random

    neg_sample = random.sample(neg, int(2*examples // 3))
    pos_sample = random.sample(pos, examples // 3)

    samples = pos_sample + neg_sample
    # random.shuffle(samples)
    prompt = (
        "Your task is to rephrase a robotic instruction into a form the robot can better understand and execute. \n"
        "You have gathered extensive data to help you better understand "
        "what language instructions the robot can successfully solve, "
        ". Pay attention to the specific language that the successful instructions use. \n"
        "Here is the data:\n"
    )
    if plain:
        pt = ""
        for inst, success in samples:
            pt += f"Instruction: {inst} | outcome: {'success' if success else 'fail'}\n\n"
        return pt
    for inst, success in samples:
        prompt += f"Instruction: {inst} | outcome: {'success' if success else 'fail'}\n\n"
    prompt += (
        "\n Now learn from this data, and think about the characteristics in the instruction that might make the robot succeed."
        f"Then rephrase the following instruction: {instruction}, but make sure not to lose any important descriptors or adjectives. Return it as a json in the form {{'rephrased_instruction': <rephrased_instruction>}}."
    )
    return prompt