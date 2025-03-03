import torch
import torch.nn as nn
from soarm100.curate_data.models.resnet import ResNet18
from soarm100.curate_data.models.core import Concatenate,MLP

class BetaVAEStates(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()

        self.encoder_state_image1 = ResNet18(in_channels=3,
                                num_kp=64,
                                spatial_coordinates=False,
                                use_clip_stem=False)
        self.encoder_state_image2 = ResNet18(in_channels=3,
                                num_kp=64,
                                spatial_coordinates=False,
                                use_clip_stem=False)
        
        total_dim = 128 + 128 + 10 * 6

        self.mlp = MLP(in_features=total_dim,
                       hidden_dims=[1024, 1024],
                       activate_final=True,
                       use_layer_norm=False,
                       dropout_rate=None)

        self.concatenate = Concatenate(model=self.mlp, flatten_time=True)

    def forward(self, x):
        '''
        x = {
            "state_image_1": state_image_1,
            "state_image_2": state_image_2,
            "state_joints": state_joints,
        }
        '''
        pass


class BetaVaeModel(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()

        self.encoder_state_image1 = ResNet18(in_channels=3,
                                num_kp=64,
                                spatial_coordinates=False,
                                use_clip_stem=False)


class VAEEncoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()

        self.encoder_state_image1 = ResNet18(in_channels=3,
                                num_kp=64,
                                spatial_coordinates=False,
                                use_clip_stem=False)


class MultiEncoderStates(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()

        self.encoder_state_image1 = ResNet18(in_channels=3,
                                            num_kp=64,
                                            spatial_coordinates=False,
                                            use_clip_stem=False)

        self.encoder_state_image2 = ResNet18(in_channels=3,
                                            num_kp=64,
                                            spatial_coordinates=False,
                                            use_clip_stem=False)

        total_dim = 128 + 128 + 10 * 6

        self.mlp = MLP(in_features=total_dim,
                       hidden_dims=[1024, 1024],
                       activate_final=True,
                       use_layer_norm=False,
                       dropout_rate=None)

        self.concatenate = Concatenate(model=self.mlp, flatten_time=True)

    def forward(self, batch):
        pass
        
        