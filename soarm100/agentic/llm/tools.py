from typing import Optional
from langchain.tools import BaseTool
from soarm100.agentic.robot import SOARM100AgenticPolicy

class PutMarkerInBoxTool(BaseTool):

    name: str = "poner_marcador_en_caja"
    description: str = "Esta herramienta se usa para poner el marcador en la caja."
    robot_policy: Optional[SOARM100AgenticPolicy] = None

    def __init__(self, robot_policy:SOARM100AgenticPolicy):
        super().__init__()
        self.robot_policy = robot_policy

    def _run(self, *args, **kwargs) -> str:
        self.robot_policy.change_policy("put_marker_in_box")
        print("PutMarkerInBoxTool")
        return "Poniendo el marcador en la caja"
    
    def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class StopRobotTool(BaseTool):

    name: str = "detener_robot"
    description: str = "Esta herramienta se usa para detener el robot."
    robot_policy: Optional[SOARM100AgenticPolicy] = None

    def __init__(self, robot_policy:SOARM100AgenticPolicy):
        super().__init__()
        self.robot_policy = robot_policy

    def _run(self, *args, **kwargs) -> str:
        self.robot_policy.change_policy("NULL")
        print("StopRobotTool")
        return "Robot detenido"
    
    def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)
    
class TeleopRobotTool(BaseTool):

    name: str = "teleop_robot"
    description: str = "Esta herramienta se usa para teleoperar el robot. Se usa cuando te piden que le des el control del robot."
    robot_policy: Optional[SOARM100AgenticPolicy] = None
    
    def __init__(self, robot_policy:SOARM100AgenticPolicy):
        super().__init__()
        self.robot_policy = robot_policy

    def _run(self, *args, **kwargs) -> str:
        self.robot_policy.change_policy("teleop")
        return "Teleoperando el robot"