import sys
import torch
from copy import copy
from queue import Queue
import time
sys.path.append("/home/leonardo/lerobot")

from lerobot.common.robot_devices.control_configs import ControlPipelineConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_utils import sanity_check_dataset_name
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from .utils import init_config,nullcontext

POLICIES = {"NULL":None,
            "put_marker_in_box":"/home/leonardo/NONHUMAN/SO-ARM100/outputs/ckpt_test/pretrained_model"}

class SOARM100AgenticPolicy:
    
    def __init__(self,cfg: ControlPipelineConfig):
        
        print("Initializing robot...")
        robot = make_robot_from_config(cfg.robot)
        
        self.robot = robot
        self.cfg   = cfg
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.running = True
        self.policy_queue = Queue()
        sanity_check_dataset_name(cfg.control.repo_id, cfg.control.policy)
        self.dataset = LeRobotDataset.create(
            cfg.control.repo_id,
            cfg.control.fps,
            root=cfg.control.root,
            robot=robot,
            use_videos=cfg.control.video,
            image_writer_processes=cfg.control.num_image_writer_processes,
            image_writer_threads=cfg.control.num_image_writer_threads_per_camera * len(robot.cameras),
        )

        if not self.robot.is_connected:
            self.robot.connect()
            print("Robot connected")
        else:
            print("Robot already connected")

    def _run(self):
        while self.running:
            #print("Running policy: ",self.policy)
            if self.policy is not None:
                #print("Ejecutando política: ", self.policy)
                self.act()
                time.sleep(0.03)
            else:
                # Pequeña pausa para no consumir CPU innecesariamente
                time.sleep(1)

    def act(self):
        observation = self.robot.capture_observation()
        pred_action = self._predict_action(observation, self.policy, self.device)
        action = self.robot.send_action(pred_action)
        print("action sent: ",action)

    def change_policy(self, policy_name: str):
        '''Aquí debemos de cambiar el policy_name por el nombre de la política que queremos ejecutar'''
        if policy_name not in POLICIES:
            raise ValueError(f"Policy {policy_name} not found")
    
        if policy_name == "NULL":
            self.policy = None
        else:
            policy_path = POLICIES[policy_name]
            self.cfg.control.policy.path = policy_path
            self.policy = make_policy(cfg=self.cfg.control.policy,
                                      device=self.device,
                                      ds_meta=self.dataset.meta)

    def _predict_action(self,observation, policy, device, use_amp=False):
        observation = copy(observation)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
        ):
            for name in observation:
                if "image" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()
                observation[name] = observation[name].unsqueeze(0)
                observation[name] = observation[name].to(device)
            action = policy.select_action(observation)
            action = action.squeeze(0)
            action = action.to("cpu")
        return action




