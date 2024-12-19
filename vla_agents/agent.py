import requests
from collections import deque
import time
import numpy as np  
import cv2
import json
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
import pdb 
from vlm import GeminiVLM
from PIL import Image
import ast
import imageio
import threading
from utils import *

def send_instruction(instruction):
    try:
        response = requests.post(
            "http://localhost:5000/update_instruction",
            json={"instruction": instruction}
        )
        if response.status_code == 200:
            print(f"Instruction sent: {instruction}")
        else:
            print(f"Failed to send instruction: {response.status_code}")
    except Exception as e:
        print(f"Error sending instruction: {e}")

def send_feedback(status):
    try:
        response = requests.post(
            "http://localhost:5000/send_feedback",
            json={"feedback": status}
        )
        if response.status_code == 200:
            print(f"Feedback sent: {status}")
        else:
            print(f"Failed to send feedback: {response.status_code}")
    except Exception as e:
        print(f"Error sending feedback: {e}")

def get_instruction():
    try:
        response = requests.get("http://localhost:5000/get_main_instruction")
        if response.status_code == 200:
            return response.json()["instruction"]
        else:
            print(f"Failed to get instruction: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting instruction: {e}")
        return None

def get_status():
    try:
        response = requests.get("http://localhost:5000/get_status")
        if response.status_code == 200:
            return response.json()["status"]
        else:
            print(f"Failed to get status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting status: {e}")
        return None 

def get_latest_frame():
    try:
        response = requests.get("http://localhost:5000/get_latest_frame")
        if response.status_code == 200:
            # Convert response content (JPEG bytes) to numpy array
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        else:
            print(f"Failed to get latest frame: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting latest frame: {e}")
        return None

def upload_video(filepath, cache=True):
    if cache:
        for file in genai.list_files():
            if filepath.split('/')[-1] == file.display_name:
                return file
    
    video_file = genai.upload_file(path=filepath)
    while video_file.state.name == "PROCESSING":
        time.sleep(0.5)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)

    return video_file

def make_prompt(inst):
    return f"""The robot was given the task of {inst}. Carefully analyze this video, 
    and evaulate the current status of the robot's task. First tell me the objects of 
    interest given the instruction, then tell me what the robot does in this video. 
    Then decide whether the robot has succeeded, failed or is still working on it. 
    Return a json in the form {{'status': <success, fail, ok>}}"""



def make_icl_video_prompt():
    instruction1 = 'put the barbaque sauce in the basket'
    video_file1 = upload_video('rollouts/icl/f1.mp4')
    first_resp = "The main object of interest is the barbaque sauce. I see the robot smoothly reach down to pick up the barbaque sauce, and then it lifts it off the ground and towards the basket but the video stops. The robot looks like it is on the right track but the video is not long enough to determine success {'status': 'ok'}"

    instruction2 = "open the middle drawer of the cabinet"
    video_file2 = upload_video('rollouts/icl/f2.mp4')
    second_resp = "The main object of interest is the cabinet and its handle. The robot starts with its gripper around the handle, and it looks like it is on track, but then it abruptly falls back and towards the ground and degenerates. {'status': 'fail'}"

    instruction3 = 'put the bowl onto the stove'
    video_file3 =  upload_video('rollouts/icl/f3.mp4')
    third_resp = "The main object is the bowl. I see the robot smoothly reach down to pick up the bowl, and then it moves it right over to the stove and drops it on the stove, a clear successs {'status': 'success'}"


    instruction4 = 'move the plate in front of the stove'
    video_file4 =  upload_video('rollouts/icl/f4.mp4')
    fourth_resp = "The main object is the plate. I see the robot is touching the plate the whole time and sliding it over. It looses its grip at the end but it looks like it is still attempting to recover {'status': 'ok'}"

    return [
    # {
    #     "role": "user",
    #     "parts": [
    #     video_file1, make_prompt(instruction1),
    #     ],
    # },
    {
        "role": "model",
        "parts": [
        first_resp,
        ],
    },
    {
        "role": "user",
        "parts": [
        video_file2, make_prompt(instruction2),
        ],
    },
    {
        "role": "model",
        "parts": [
        second_resp,
        ],
    },
    {
        "role": "user",
        "parts": [
        video_file3, make_prompt(instruction3),
        ],
    },
    {
        "role": "model",
        "parts": [
        third_resp,
        ], 
    },
    {
        "role": "user",
        "parts": [
        video_file4, make_prompt(instruction4),
        ],
    },
    # {
    #     "role": "model",
    #     "parts": [
    #     fourth_resp,
    #     ],
    # },
    ]
    

class Feedback():

    def __init__(self, model, buffer_len=130):
        self.vlm = GeminiVLM(model=model, system_instruction="You are an assistant whose purpose is to watch a video of a robot as it attempts to perform a task, and determine whether the robot has succeeded, failed, or is still in progress. Note that for tasks like placing objects somewhere, if it looks like the robot is ABOUT to place the object in the right location, it should count as a success. You should only return 'fail' when it is clear the robot has stalled and is not moving at all OR if it is moving nonsensically.")
        self.frames = deque(maxlen=buffer_len)
        self.icl = make_icl_video_prompt()

        self.running = True  # To control the frame update thread
        self.lock = threading.Lock()  # Ensure thread safety when resetting frames

    def start(self):
        self.update_thread = threading.Thread(target=self._update_frames, daemon=True)
        self.update_thread.start()

    def _update_frames(self):
        """Runs in a separate thread to continuously collect frames."""
        while self.running:
            t = time.time()
            t = time.time()
            frame = get_latest_frame()
            status = get_status()
            if status == 'waiting':
                with self.lock:
                    self.running = False
                    self.frames.clear()
                break

            with self.lock:
                self.frames.append(frame)
            time.sleep(0.3)

    def reset(self):
        """Clears the frame buffer."""
        with self.lock:  # Use the lock to ensure thread-safe operations
            self.frames.clear()
        print("Frame buffer has been reset.")

    def get_feedback(self, instruction):
        # Returns success, fail, or ok
        print("Starting feedback process")
        t = time.time()
        with self.lock:
            assert len(self.frames) > 0, "Frame buffer is empty!"  # Check buffer size safely
            frames_snapshot = list(self.frames)  # Copy the frames to avoid mutation issues
        try:
            self.vlm.reset()
            path = self.save_rollout_video(frames_snapshot)
            self.vlm.session.history = self.icl
            prompt = f"The robot was given the task of {instruction}. Carefully analyze this video, and evaluate the current status of the robot's task. First tell me the objects of interest given the instruction, then tell me what the robot does in this video, and lastly return a json in the form {{'status': <success, fail, ok>}}"
            video_file = upload_video(path, cache=False)
            response = self.vlm.call_video(video_file, prompt)
            eval_resp = ast.literal_eval(response[response.rindex("{"):response.rindex("}") + 1])
            prediction = eval_resp["status"]
        except Exception as e:
            print(e)
            prediction = "ok"
            response = ""
        print(f"Feedback took {time.time() - t} seconds, returning {prediction}")
        print('\n\n', response)
        return prediction

    def save_rollout_video(self, rollout_images):
        """Saves an MP4 replay of an episode."""
        mp4_path = f"temp/frames.mp4"
        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in rollout_images:
            # Convert RGB to BGR for imageio
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.append_data(img_bgr)
        video_writer.close()
        return mp4_path

    def stop(self):
        """Stops the frame update thread."""
        self.running = False
        self.update_thread.join()

class Agent():

    def __init__(self, model):
        self.vlm = GeminiVLM(model=model, system_instruction="You are an assistant that has the ability to give language commands to a robot. Your purpose is to assist human requests by controlling this robot, and updating your commands based on feedback you recieve. ")
        self.starting_instruction = None

    def generate_instruction(self, starting=False, update=None):
        if starting:
            prompt = (
                f"The human is giving you the following (broad) instruction: {self.starting_instruction}. "
                f"Based on this instruction and the image of the scene, generate a sub-instruction for the robot to execute, which can be just part of the original instruction if the original instruction is complex. "
                f"From your past experience with this robot, you have collected the following data about successful and unsuccessful instructions: "
                f"{make_icl_instructions('', '', plain=True)}\n"
                f"Analyze the characteristics and patterns of successful and failed instructions. "
                f"Briefly explain your reasoning and then return the command in JSON format: {{\"command\": <command>}}"
            )
        else:
            assert update is not None
            prev_instruction = update['instruction']
            outcome = update['outcome']
            prompt = (
                f"As a reminder, your ultimate goal is to get the robot to complete the following (broad) instruction: {self.starting_instruction}. "
                f"You just recieved feedback that the robot {'successfully completed the' if outcome == 'success' else 'failed to complete the'} previous command {prev_instruction}'."
                f"Use this information as well as the image of the scene, and your past experience with the robot to briefly reflect on how your commands have performed so far, and plan what you should do next to make progress towards your end goal. "
                f"Briefly explain your thinking and then return the command in the json {{'command': <command>}}"
            )

        for attempt in range(4):
            try:
                frame = get_latest_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                response = self.vlm.call_chat(1000, [frame], prompt)
                instruction = ast.literal_eval(response[response.rindex("{"):response.rindex("}") + 1])['command']
                print(f'Response: {response}')
                response = response[0: response.rindex("{")]
                break
            except Exception as e:
                if attempt == 3:
                    raise Exception("Failed to generate instruction after 4 attempts") from e
                print(f"Attempt {attempt + 1} failed. Retrying...")

        
        return response, instruction        

    def run_episode(self):
        feedback = Feedback('gemini-1.5-flash')
        while True:
            instruction = get_instruction()
            if instruction is not None and instruction != 'default':
                print(f"Received starting instruction: {instruction}")
                self.starting_instruction = instruction
                response, instruction = self.generate_instruction(starting=True)
                send_instruction(instruction) 
                break
            time.sleep(0.5)

        feedback.start()


        t = time.time()
        last_feedback = t - 4
        while True: 
            time.sleep(0.3)
            dt = time.time()
            with feedback.lock: 
                if not feedback.running:
                    print('recieved environment finished signal')
                    return
                l = len(feedback.frames)

            if (dt - last_feedback) > 13 and l > 120:
                last_feedback = time.time()
                send_feedback('Starting feedback calculation')
                status = feedback.get_feedback(instruction)
                with feedback.lock:
                    if not feedback.running:
                        return
                send_feedback(status)
                if status != 'ok':
                    time.sleep(0.3)
                    send_instruction("reset")
                    time.sleep(1)
                    response, new_instruction = self.generate_instruction(starting=False, update= {"instruction": instruction, "outcome": status})
                    with feedback.lock:
                        if not feedback.running:
                            return
                    send_instruction(new_instruction)
                    time.sleep(0.5)
                    send_feedback(' ')
                    instruction = new_instruction
                    feedback.reset()

    def run(self):
        while True:
            self.run_episode()
            self.vlm.reset()
            
if __name__ == "__main__":
    agent = Agent("gemini-1.5-pro")
    agent.run()
