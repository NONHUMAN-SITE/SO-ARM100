from typing import Optional
from langchain.tools import BaseTool
from soarm100.agentic.robot import SOARM100AgenticPolicy

class PutMarkerInBoxTool(BaseTool):

    name: str = "put_marker_in_box"
    description: str = "This tool is used to put the marker in the box."
    robot_policy: Optional[SOARM100AgenticPolicy] = None

    def __init__(self, robot_policy:SOARM100AgenticPolicy):
        super().__init__()
        self.robot_policy = robot_policy

    def _run(self, *args, **kwargs) -> str:
        self.robot_policy.change_policy("put_marker_in_box")
        print("PutMarkerInBoxTool")
        return "Putting the marker in the box"
    
    def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class StopRobotTool(BaseTool):

    name: str = "stop_robot"
    description: str = "This tool is used to stop the robot."
    robot_policy: Optional[SOARM100AgenticPolicy] = None

    def __init__(self, robot_policy:SOARM100AgenticPolicy):
        super().__init__()
        self.robot_policy = robot_policy

    def _run(self, *args, **kwargs) -> str:
        self.robot_policy.change_policy("NULL")
        print("StopRobotTool")
        return "Robot stopped"
    
    def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)
    
class TeleopRobotTool(BaseTool):

    name: str = "teleop_robot"
    description: str = "This tool is used to teleoperate the robot. It is used when you are asked to give control to the robot."
    robot_policy: Optional[SOARM100AgenticPolicy] = None
    
    def __init__(self, robot_policy:SOARM100AgenticPolicy):
        super().__init__()
        self.robot_policy = robot_policy

    def _run(self, *args, **kwargs) -> str:
        self.robot_policy.change_policy("teleop")
        return "Teleoperating the robot"