import time
import sys
import pathlib
path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(path))
from soarm100.agentic.client import Gr00tRobotInferenceClient

def main():
    host = "rnpvi-132-251-2-3.a.free.pinggy.link"
    port = 39683
    client = Gr00tRobotInferenceClient(host=host, port=port)
    print("Connected to server")
    while True:
        print("Pinging server")
        response = client.ping()
        print(response)
        time.sleep(1)

if __name__ == "__main__":
    main()