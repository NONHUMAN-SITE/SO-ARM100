import time
import torch
from functools import partial
import numpy as np
import threading
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from soarm100.agentic.client import Gr00tRobotInferenceClient
from soarm100.agentic.llm.tools import GIVE_INSTRUCTIONS_TOOL, GET_FEEDBACK_TOOL
from soarm100.agentic.llm.agent import SupervisorAgent
from soarm100.agentic.realtime import RealtimeAudioChat
from soarm100.agentic.robot import SO100Robot
from soarm100.agentic.utils import view_img
from soarm100.logger import logger


USE_VLM = True
VLM_NAME = "gemini"  # openai, gemini
ACTIONS_TO_EXECUTE = 10
ACTION_HORIZON = 16
MODALITY_KEYS = ["single_arm", "gripper"]
CAM_IDX = 1  # The camera index


def run_robot(client_instance:Gr00tRobotInferenceClient, robot_instance:SO100Robot):
    with robot_instance.activate():
        while True:
            logger.log(f"Robot is running", level="success")
            if client_instance.get_lang_instruction() == "Stay quiet":
                robot_instance.go_home()
                continue
            for i in range(ACTION_HORIZON):
                img = robot_instance.get_current_img()
                view_img(img)
    
                state = robot_instance.get_current_state()
                action = client_instance.get_action(img,state)
    
                for j in range(ACTION_HORIZON):
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][j]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    assert concat_action.shape == (6,), concat_action.shape
                    robot_instance.set_target_state(torch.from_numpy(concat_action))
                    time.sleep(0.015)
                    # get the realtime image
                    img = robot_instance.get_current_img()
                    view_img(img)
            time.sleep(0.015)

def callback_give_instructions(instruction:str,client_instance:Gr00tRobotInferenceClient,):
    logger.log(f"Received instruction: {instruction}", level="success")
    client_instance.set_lang_instruction(instruction)
    return "Executing instructions. In progress..."

def callback_get_feedback(consult_query:str,task:str,robot_instance:SO100Robot,supervisor_agent:SupervisorAgent,):
    logger.log(f"Received consult query: {consult_query}", level="success")
    logger.log(f"Received task: {task}", level="success")
    images = [robot_instance.get_current_img()]
    logger.log(f"Images: {images}", level="success")
    prompt = f"""
    Consult Query: {consult_query}
    Task: {task}
    """
    response = supervisor_agent.analyze(images,prompt)
    logger.log(f"Response Supervisor Agent: {response}", level="success")
    return response

if __name__ == "__main__":

    HOST = "127.0.0.1"
    PORT = 3000

    logger.log("Starting the robot client", level="info")

    robot_client = SO100Robot(enable_camera=True,cam_idx=2)

    supervisor_agent = SupervisorAgent()

    logger.log("Starting the robot client...", level="info")

    client = Gr00tRobotInferenceClient(host=HOST, port=PORT,language_instruction="Stay quiet")

    partial_callback_give_instructions = partial(callback_give_instructions, client_instance=client)
    partial_callback_get_feedback = partial(callback_get_feedback, robot_instance=robot_client, supervisor_agent=supervisor_agent)

    CALLBACK_DICT = {
        "give_instructions": partial_callback_give_instructions,
        "get_feedback": partial_callback_get_feedback
    }

    logger.log("Starting the realtime audio chat...", level="info")

    realtime = RealtimeAudioChat(tools=[GIVE_INSTRUCTIONS_TOOL,
                                        GET_FEEDBACK_TOOL],
                                 callback_dict=CALLBACK_DICT)
    
    logger.log("Starting the robot thread", level="info")
    robot_thread = threading.Thread(target=run_robot, args=(client, robot_client))
    robot_thread.daemon = True
    robot_thread.start()

    logger.log("Starting the realtime audio chat", level="info")
    realtime.start()

    try:
        while True:
            if not robot_thread.is_alive():
                logger.log("Robot thread has stopped unexpectedly", level="error")
                break
            if not realtime.is_active():
                logger.log("Realtime chat has stopped unexpectedly", level="error")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.log("Received interrupt signal. Shutting down...", level="info")
    finally:
        realtime.stop()
        logger.log("System shutdown complete", level="info")

