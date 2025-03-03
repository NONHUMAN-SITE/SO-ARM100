import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import torch
from soarm100.curate_data.dataset import (create_dataset,
                                          collate_fn_mutual_information)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from soarm100.curate_data.config import CurateDataConfig
from soarm100.curate_data.models.beta_vae import (BetaVAES,
                                                  MultiEncoderStates)
'''
El dataset tiene el siguiente formato:

observation.images.laptop: torch.Size([3, 480, 640])
observation.images.phone: torch.Size([3, 480, 640])
action: torch.Size([100, 6])
observation.state: torch.Size([6])
timestamp: torch.Size([])
frame_index: 76 torch.Size([]) #Index relativo al episodio
episode_index: 0 torch.Size([]) #Index del episodio
index: 76 torch.Size([]) #Index general del dataset
task_index: 0 torch.Size([]) ¿?
action_is_pad: tensor([False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False]) torch.Size([100])
        Hace padding para solo procesar acciones válidase
task: string
'''

@parser.wrap()
def train_vaes(cfg: CurateDataConfig):
    #cfg.validate()

    dataset = create_dataset(cfg,type_model=cfg.type)

    print(cfg.type)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.batch_size,
                                             collate_fn=lambda x: collate_fn_mutual_information(x,cfg.type))
    
    z_dim = {
        "states": 16,
        "actions": 16,
    }[cfg.type]

    if cfg.type == "states":
        model = BetaVAES(encoder=MultiEncoderStates(),
                         z_dim=z_dim)
    else:
        model = BetaVAES(encoder=MultiEncoderStates(),
                         z_dim=z_dim)

    for batch in dataloader:
        print(batch.keys())
        mean, logvar = model(batch)
        print(mean.shape, logvar.shape)
        break
        
        
    




if __name__ == "__main__":
    train_vaes()

