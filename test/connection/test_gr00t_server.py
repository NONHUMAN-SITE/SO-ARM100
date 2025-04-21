import sys
import pathlib
path = pathlib.Path(__file__).parent.parent.parent
from gr00t.experiment.data_config import So100DataConfig
from gr00t.model.policy import Gr00tPolicy
sys.path.append(str(path))
from soarm100.agentic.server import RobotInferenceServerTCP

def main():
    host = "127.0.0.1"
    port = 3000

    #Model config
    data_config = So100DataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    model_path = "/home/lperez/so100-checkpoints/checkpoint-1500"
    embodiment_tag = "new_embodiment"
    denoising_steps = 4
    
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
    )


    server = RobotInferenceServerTCP(model=policy, host=host, port=port)
    server.run()

if __name__ == "__main__":  
    main()