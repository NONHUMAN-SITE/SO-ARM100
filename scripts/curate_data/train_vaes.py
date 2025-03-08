import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import torch
import wandb
from tqdm import tqdm   
from lerobot.configs import parser
from soarm100.curate_data.dataset import create_dataset
from soarm100.curate_data.config import CurateDataConfig
from soarm100.curate_data.models.beta_vae import (BetaVAE,
                                                  MultiEncoderStates,
                                                  MultiDecoderStates)
from soarm100.curate_data.utils import (cycle,
                                        make_normalization_layers,
                                        make_optimizer_and_scheduler)
from soarm100.logger import logger

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = create_dataset(cfg,type_model=cfg.type,device=device)

    normalize_targets, normalize_inputs, unnormalize_targets = make_normalization_layers(dataset,device=device)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.batch_size,
                                             num_workers=cfg.num_workers,
                                             shuffle=True,
                                             sampler=None,
                                             pin_memory=True,
                                             drop_last=False)
    
    normalizer = normalize_inputs if cfg.type == "states" else normalize_targets

    dl_iter = cycle(dataloader)

    z_dim = {
        "states": 16,
        "actions": 16,
    }[cfg.type]

    mlp_dim = {
        "states": 1024,
        "actions": 1024,
    }[cfg.type]

    weights = [1.0, 1.0, 1.0]  # Peso por cada modalidad

    if cfg.type == "states":
        model = BetaVAE(encoder=MultiEncoderStates(),
                        decoder=MultiDecoderStates(z_dim=z_dim,mlp_dim=mlp_dim),
                        z_dim=z_dim,
                        weights=weights).to(device)
    else:
        model = BetaVAE(encoder=MultiEncoderStates(),
                        decoder=MultiDecoderStates(z_dim=z_dim,mlp_dim=mlp_dim),
                        z_dim=z_dim,
                        weights=weights).to(device)

    optim, scheduler = make_optimizer_and_scheduler(model,cfg)

    wandb.init(project="soarm100-mutual-information-{}".format(cfg.type),
               name=cfg.type)   

    output_dir = os.path.join(cfg.output_dir,cfg.type)

    os.makedirs(output_dir,exist_ok=True)

    for step in tqdm(range(1,cfg.steps+1),total=cfg.steps):
        batch = next(dl_iter)
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        batch = normalizer(batch)
        loss, info = model.train_step(batch, optim)
        scheduler.step()
        wandb.log({"loss": loss, "info": info},step=step)
        if step % cfg.log_freq == 0:
            logger.log(loss=loss,info=info,lr=scheduler.get_last_lr()[0])
        if step % cfg.save_freq == 0:
            model.save_checkpoint(os.path.join(output_dir,f"model_{step}.pt"))

if __name__ == "__main__":
    train_vaes()

