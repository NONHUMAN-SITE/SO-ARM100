import sys
import pathlib
path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(path))
from soarm100.agentic.server import RobotInferenceServerTCP

def main():
    host = "127.0.0.1"
    port = 3000
    server = RobotInferenceServerTCP(host=host, port=port)
    server.run()

if __name__ == "__main__":  
    main()