import torch
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.configs.types import FeatureType, NormalizationMode
from soarm100.curate_data.dataset import MutualInformationDataset


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def make_normalization_layers(dataset: MutualInformationDataset,device: torch.device):
    
    '''
    En features es donde se define el tipo de dato que se va a normalizar; es decir, que define
    un diccionario con los tipos de datos que se van a normalizar. Ejemplo:

    {'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(6,)),
    'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,)),
    'observation.images.laptop': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)),
    'observation.images.phone': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640))}
    
    Entonces si quisieras cambiarlo, solo deberías de cambiar el diccionario de features.
    '''

    features = dataset_to_policy_features(dataset.meta().features)

    dataset_stats = dataset.meta().stats

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    normalization_mapping = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }

    normalize_targets = Normalize(output_features,
                                  normalization_mapping,
                                  dataset_stats).to(device)
    
    normalize_inputs = Normalize(input_features,
                                 normalization_mapping,
                                 dataset_stats).to(device)
    
    unnormalize_targets = Unnormalize(output_features,
                                      normalization_mapping,
                                      dataset_stats).to(device)

    return (normalize_targets,
            normalize_inputs,
            unnormalize_targets)


def make_optimizer_and_scheduler(model,cfg):
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                           T_max=cfg.steps,
                                                           eta_min=cfg.learning_rate/100)
    return optim, scheduler
