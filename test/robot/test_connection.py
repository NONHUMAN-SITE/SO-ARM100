import time
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from soarm100.agentic.robot import SO100Robot

# Note: The calibration folder needs to stay in the .cache/calibration/so100 in the same directory as the script

if __name__ == "__main__":
    robot = SO100Robot()
    robot.connect()
    time.sleep(5)
    robot.disconnect()
