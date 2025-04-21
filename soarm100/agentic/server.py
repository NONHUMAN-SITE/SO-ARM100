from dataclasses import dataclass
from io import BytesIO
from typing import Callable
import torch
import zmq
import asyncio
import websockets
import json
from gr00t.model.policy import BasePolicy
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import So100DataConfig
from gr00t.model.policy import Gr00tPolicy


class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServerTCP:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                print(f"Sending result: {result}")
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(b"ERROR")

class RobotInferenceServerTCP(BaseInferenceServerTCP):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model = None, host: str = "*", port: int = 5555):
        super().__init__(host, port)
        if model is not None:
            self.register_endpoint("get_action", model.get_action)
            self.register_endpoint(
                "get_modality_config", model.get_modality_config, requires_input=False
            )

    @staticmethod
    def start_server(policy: BasePolicy, port: int):
        server = RobotInferenceServerTCP(policy, port=port)
        server.run()


class BaseInferenceServerWebSocket:
    """
    An inference server that uses WebSockets and listens for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._endpoints: dict[str, EndpointHandler] = {}
        self.running = True

        # Register default endpoints
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        self.running = False
        return {"status": "ok", "message": "Server is shutting down"}

    def _handle_ping(self):
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    async def _handle_connection(self, websocket):
        async for message in websocket:
            try:
                request = json.loads(message)
                endpoint = request.get("endpoint", "get_action")
                data = request.get("data", {})

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                if handler.requires_input:
                    response = handler.handler(data)
                else:
                    response = handler.handler()

                await websocket.send(json.dumps(response))
            except Exception as e:
                print(f"Error in handler: {e}")
                await websocket.send(json.dumps({"status": "error", "message": str(e)}))

    async def run(self):
        print(f"WebSocket server starting at ws://{self.host}:{self.port}")
        async with websockets.serve(self._handle_connection, self.host, self.port):
            while self.running:
                await asyncio.sleep(0.1)
        print("WebSocket server stopped.")

class RobotInferenceServerWebSocket(BaseInferenceServerWebSocket):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "localhost", port: int = 8765):
        super().__init__(host, port)
        self.register_endpoint("get_action", model.get_action)
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )

    @staticmethod
    def start_server(policy: BasePolicy, host: str = "localhost", port: int = 8765):
        server = RobotInferenceServerWebSocket(policy, host=host, port=port)
        asyncio.run(server.run())


if __name__ == "__main__":

    '''
    To Do:
    - Save a checkpoint example of the model just for testing
    - Create an endpoint with a tcp connection to the real robot using pinggy.io/
    '''

    model_path = "/home/jiajun/workspace/gr00t/models/soarm100/isaac_groot/model.pth"
    host = "0.0.0.0"
    port = 5555

    # Config model
    data_config = So100DataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    embodiment_tag = "new_embodiment" #don't change this
    denoising_steps = 4

    model = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
    )

    # Run server
    server = RobotInferenceServerTCP(model=model, host=host, port=port)
    server.run()