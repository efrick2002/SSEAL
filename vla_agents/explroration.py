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
# from threading import concurrent
import concurrent

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

def make_prompt_success(inst):
    first_prompt = f"The robot was given the task of {inst}. Carefully analyze this video, and evaulate if the robot has at this point succeeded in this task. First tell me what the robot does in this video, and lastly return a json in the form {{'success': <1 or 0>}}"
    return first_prompt

def make_icl_video_prompt_success():
    instruction1 = 'open the middle drawer of the cabinet'
    video_file1 = upload_video('/home/dylangoetting/toolvla/rollouts/2024_12_07/2024_12_07-01_35_44--episode=1--success=True--task=open_the_middle_drawer_of_the_cabinet.mp4')
    first_resp = "I see the robot smoothly move towards the middle drawer of the cabined, and slighly closes its gripper around the middle handle. It pulls it back . {'success': 1}"

    video_file2 = upload_video('/home/dylangoetting/toolvla/rollouts/2024_12_07/2024_12_07-01_55_33--episode=1--success=False--task=open_the_middle_drawer_of_the_cabinet.mp4')
    second_resp = "I see the robot smoothly approach the middle handle of the cabinet, but suddenly it seems to fall down and then it moves its gripper to point towards the camera. From this video, the robot has not completed its task. {'success': 0}"

    instruction3 = 'pick up the alphabet soup and place it in the basket'
    video_file3 =  upload_video('/home/dylangoetting/toolvla/rollouts/2024_12_08/2024_12_08-18_26_51--episode=1--success=True--task=pick_up_the_alphabet_soup_and_place_it_in_the_bask.mp4')
    third_resp = "I see the robot smoothly reach down to pick up the alphabet soup, and then it moves it right over to the basket. It is close enough to be inside the basket, thus the robot has succeeded. {'success': 1}"

    return [
    {
        "role": "user",
        "parts": [
        video_file1, make_prompt_success(instruction1),
        ],
    },
    {
        "role": "model",
        "parts": [
        first_resp,
        ],
    },
    {
        "role": "user",
        "parts": [
        video_file2, make_prompt_success(instruction1),
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
        video_file3, make_prompt_success(instruction3),
        ],
    },
    {
        "role": "model",
        "parts": [
        third_resp,
        ],
    }
    ]

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
    {
        "role": "user",
        "parts": [
        video_file1, make_prompt(instruction1),
        ],
    },
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
    {
        "role": "model",
        "parts": [
        fourth_resp,
        ],
    },
    ]
    

class Feedback():

    def __init__(self, model, buffer_len=120):
        self.vlm = GeminiVLM(model=model, system_instruction="You are an assistant whose purpose is to watch a video of a robot performing a task, and detetermine whether the robot has succeeded or not. ")
        self.frames = deque(maxlen=buffer_len)
        self.icl = make_icl_video_prompt_success()

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
            prompt = make_prompt_success(instruction)
            video_file = upload_video(path, cache=False)
            response = self.vlm.call_video(video_file, prompt)
            eval_resp = ast.literal_eval(response[response.rindex("{"):response.rindex("}") + 1])
            prediction = eval_resp["success"]
        except DeprecationWarning as e:
            print(e)
            prediction = 0
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
        self.vlm = GeminiVLM(model=model, system_instruction="You are an assistant that has the ability to give language commands to a robot. Your purpose is to assist human requests by controlling this robot. Your goal is to figure out what types of tasks the robot can perform, and also the language styles that work best for the robot.")
        self.starting_instruction = None

    def generate_instruction(self, starting=False, update=None):
        prompt = (
            "Observe the image, which shows the robot and current state of the scene. Your task is to figure out the language style and phrasing that enables the robot to best perform its tasks. "
            "You are currently in the EXPLORATION PHASE. This means you should try to explore different tasks that the robot can perform, and especially "
            "the different language styles in your command. First, generate a list of around ~10 language instructions for the robot to attempt, which should "
            "vary both the task and language style. First tell me what you see in the scene, then return a python list with each instruction in quotes, separated by commas. "
        )


        for attempt in range(4):
            try:
                response = self.vlm.call([get_latest_frame()], prompt)
                instructions = ast.literal_eval(response[response.rindex("["):response.rindex("]") + 1])
                print(f'Response: {response}')
                break
            except Exception as e:
                if attempt == 3:
                    raise Exception("Failed to generate instruction after 4 attempts") from e
                print(f"Attempt {attempt + 1} failed. Retrying...")

        
        return response, instructions        

    def run_exploration(self):
        logs = []
        response, instructions = self.generate_instruction(starting=True)
        for string in instructions:
            send_instruction('reset hard')
            time.sleep(3)
            feedback = Feedback('gemini-2.0-flash-exp', buffer_len=140)
            feedback.start()    
            send_instruction(string)
            t = time.time()
            while True:
                with feedback.lock:
                    l = len(feedback.frames)
                if (time.time() - t) > 22 and l > 130:
                    status1 = feedback.get_feedback(string)
                    logs.append({'instruction': string, 'outcome': status1})
                    break
                time.sleep(0.3)
            feedback.stop()

        with open('exploration_logs.json', 'r+') as file:
            existing_logs = json.load(file)
            existing_logs.extend(logs)
            file.seek(0)
            json.dump(existing_logs, file, indent=4)
            file.truncate()


    def run(self):
        self.run_exploration()
        send_instruction('next')
        self.run_exploration()
        send_instruction('next')
            
if __name__ == "__main__":
    agent = Agent("gemini-1.5-pro")
    agent.run()
