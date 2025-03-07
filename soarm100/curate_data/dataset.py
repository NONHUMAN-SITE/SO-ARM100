import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.utils.random_utils import set_seed
from soarm100.curate_data.config import CurateDataConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class MutualInformationDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset: LeRobotDataset,
                 type_model: str,
                 device: torch.device,
                 action_chunks_h:int = 10):
        
        self.dataset = dataset
        self.action_chunks_h = action_chunks_h
        self.type_model = type_model
        self.device = device
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        '''
        Tener cuidado con el action_pading, debemos de ver
        cómo funciona este y cómo podemos modificar el action_chunk
        para esto.
        '''
        timestep = self.dataset[index]

        if self.type_model == "states":
            state_image_1 = timestep["observation.images.laptop"] # [3, 480, 640]
            state_image_2 = timestep["observation.images.phone"] # [3, 480, 640]
            state_joints = timestep["observation.state"] # [6]

            # En caso queremos cambiar los keys, podemos hacerlo aquí
            return {
                "observation.images.laptop": state_image_1,
                "observation.images.phone": state_image_2,
                "observation.state": state_joints,
            }

        elif self.type_model == "actions":

            '''
            Debemos de tener cuidado con el padding de los actions.
            '''

            action = timestep["action"] # [100, 6]            
            action_joints_chunk = self._get_action_chunks(action) # [10, 6]

            # En caso queremos cambiar los keys, podemos hacerlo aquí
            return {
                "action": action_joints_chunk,
            }

    def _get_action_chunks(self, action: torch.Tensor):
        '''
        Devuelve chunks de acciones de tamaño action_chunks_h
        '''
        return action[:self.action_chunks_h,:]

    def meta(self):
        return self.dataset.meta


def create_dataset(cfg: CurateDataConfig,type_model: str,device: torch.device):

    if type_model not in ["states", "actions"]:
        raise ValueError(f"type_model must be one of ['states', 'actions'] got {type_model}")

    if cfg.seed is not None:
        set_seed(cfg.seed)

    offline_dataset = make_dataset(cfg)

    return MutualInformationDataset(offline_dataset,type_model,device)

if __name__ == "__main__":
    create_dataset()
