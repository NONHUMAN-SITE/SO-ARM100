import time
import sys
import pathlib
path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(path))
from soarm100.agentic.client import Gr00tRobotInferenceClient

def main():
    host = "rnter-45-231-80-103.a.free.pinggy.link  "
    port = 33685
    client = Gr00tRobotInferenceClient(host=host, port=port)
    print("Connected to server")
    while True:
        print("Pinging server")
        action = client.sample_action()
        print(action)
        time.sleep(1)

if __name__ == "__main__":
    main()