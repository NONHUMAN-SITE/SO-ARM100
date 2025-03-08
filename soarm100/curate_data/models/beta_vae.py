import torch
import torch.nn as nn
from typing import Dict,Tuple
import sys
sys.path.append("/home/lperez/main/NONHUMAN/SO-ARM100")
from soarm100.curate_data.models.resnet import ResNet18,ResNet18Decoder
from soarm100.curate_data.models.core import Concatenate,MLP
import torch.nn.functional as F

class BetaVAE:
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 z_dim: int,
                 type: str,
                 beta: float = 1.0,
                 weights: list[float] | None = None):
        """
        encoder: Módulo base para procesar la entrada.
        decoder: Módulo para reconstruir (se espera, por ejemplo, MultiDecoderStates).
        z_dim: Dimensión del espacio latente.
        beta: Factor de peso para el término KL.
        weights: Lista de pesos (floats) para cada componente de reconstrucción
                 (se asume el orden: [state_image_1, state_image_2, state_joints]).
        """
        if type == "states":
            self.model = BetaVaeModel(
                VAEEncoderStates(encoder, z_dim),
                VAEDecoderStates(decoder)
            )
            keys_batch = ["observation.images.laptop", "observation.images.phone", "observation.state"]
        elif type == "actions":
            raise NotImplementedError("Actions are not implemented yet")
            self.model = BetaVaeModel(
                VAEEncoderActions(encoder, z_dim),
                VAEDecoderActions(decoder)
            )
            keys_batch = ["action"]
            
        if weights is None:
            self.weights = [1.0/200, 1.0/200, 1.0]
        else:   
            self.weights = weights
        self.beta = beta

    def to(self, device: torch.device) -> 'BetaVAE':
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Tuple[float, dict]:
        self.model.train()
        optimizer.zero_grad()
        x_hat, mean, logvar = self.model(batch)
        loss, info = self._loss(x_hat, mean, logvar, batch)
        loss.backward()
        optimizer.step()
        return loss.item(), info

    def val_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, dict]:
        self.model.eval()
        with torch.no_grad():
            x_hat, mean, logvar = self.model(batch)
            loss, info = self._loss(x_hat, mean, logvar, batch)
        return loss.item(), info

    def predict(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor],
                                                               torch.Tensor,
                                                               torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)

    def _loss(self,
              x_hat: Dict[str, torch.Tensor],
              mean: torch.Tensor,
              logvar: torch.Tensor,
              batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        
        # Término KL: 0.5*sum(-logvar - 1 + exp(logvar) + mean^2) por instancia.
        kl = 0.5 * torch.sum(-logvar - 1 + torch.exp(logvar) + mean.pow(2), dim=1)
        kl_loss = torch.mean(kl)
        # Pérdida de reconstrucción por cada componente.
        recon_loss_total = 0.0
        recon_loss_dict = {}
        keys = ["observation.images.laptop", "observation.images.phone", "observation.state"]
        for i, key in enumerate(keys):
            if key in batch and key in x_hat:
                # Utilizamos error cuadrático medio con reducción "mean".
                recon_loss = F.mse_loss(x_hat[key], batch[key], reduction='mean')
                weight = self.weights[i] if i < len(self.weights) else 1.0
                recon_loss_weighted = weight * recon_loss
                recon_loss_dict[key] = recon_loss.item()
                recon_loss_total += recon_loss_weighted
        loss = recon_loss_total + self.beta * kl_loss
        info = {f"recon_loss/{key}": val for key, val in recon_loss_dict.items()}
        info["recon_loss/total"] = recon_loss_total.item() if isinstance(recon_loss_total, torch.Tensor) else recon_loss_total
        info["kl_loss"] = kl_loss.item()
        info["loss"] = loss.item()
        return loss, info


class BetaVaeModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(BetaVaeModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(batch)
        stddev = torch.exp(0.5 * logvar)
        z = mean + stddev * torch.randn_like(stddev)
        x_hat = self.decoder(z,batch)
        return x_hat,mean,logvar

class VAEEncoderStates(nn.Module):
    """
    Encoder VAE que proyecta la salida del encoder base a media y log-varianza
    del espacio latente.
    """
    def __init__(self, encoder: nn.Module, z_dim: int):
        super(VAEEncoderStates, self).__init__()
        self.encoder = encoder
        
        # Capa de proyección a media y log-varianza (2 * z_dim)
        self.z_proj = nn.Linear(encoder.output_dim, 2 * z_dim)
        # Inicialización Xavier/Glorot
        nn.init.xavier_uniform_(self.z_proj.weight)
        nn.init.zeros_(self.z_proj.bias)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Codifica la entrada usando el encoder base
        x = self.encoder(x)
        
        # Proyecta a media y log-varianza
        z = self.z_proj(x)

        mean, logvar = torch.chunk(z, chunks=2, dim=-1)

        return mean, logvar

class VAEDecoderStates(nn.Module):
    """
    Decoder del VAE que utiliza un modelo (ej. MultiDecoderStates) para
    generar salidas a partir del vector latente z. Luego, para cada salida,
    se realiza una proyección (si es necesaria) para tener la forma deseada,
    es decir, se intenta que la salida coincida en forma con la observación en
    el batch (por ejemplo, la imagen u otra modalidad).

    Se asume que el modelo retorna una tupla en el siguiente orden:
      (state_image_1, state_image_2, state_joints)
    y que en el batch existen las claves correspondientes.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Se utiliza un ModuleDict para almacenar las capas de proyección para cada key.
        # 6: number of joints
        self.observation_state = nn.Linear(self.model.mlp_dim, 6)

    def _output_proj(self, x_hat: torch.Tensor, target: torch.Tensor, key: str) -> torch.Tensor:
        """
        Compara las dimensiones de x_hat y target. Si no coinciden todas las dimensiones,
        se aplica una proyección lineal para mapear x_hat a un vector de tamaño igual a
        np.prod(target.shape[matching_dims:]) y se reconfigura a target.shape.

        Se asegura de que la capa de proyección se cree en el mismo dispositivo que x_hat.
        """
        matching = 0
        for d_target, d_pred in zip(target.shape, x_hat.shape):
            if d_target == d_pred:
                matching += 1
            else:
                break
        if key == "observation.state":
            return self.observation_state(x_hat)
        else:
            return x_hat

    def forward(self, z: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Se espera que `batch` tenga al menos las keys:
           - observation.images.laptop
           - observation.images.phone
           - observation.state
        y que el modelo retorne una tupla con tres elementos correspondientes.
        """
        outputs = self.model(z, batch)
        # Se asume el siguiente orden:
        keys = ["observation.images.laptop", "observation.images.phone", "observation.state"]
        num_outputs = len(outputs)
        keys = keys[:num_outputs]
        x_hat = {k: out for k, out in zip(keys, outputs)}
        # Para cada key, se proyecta x_hat a la forma de la entrada (del batch) si es necesario.
        for key, pred in x_hat.items():
            if key in batch:
                x_hat[key] = self._output_proj(pred, batch[key], key)
        return x_hat

class MultiEncoderStates(nn.Module):
    def __init__(self):
        super(MultiEncoderStates, self).__init__()

        self.encoder_state_image1 = ResNet18(in_channels=3,
                                            num_kp=64,
                                            spatial_coordinates=False,
                                            use_clip_stem=False)

        self.encoder_state_image2 = ResNet18(in_channels=3,
                                            num_kp=64,
                                            spatial_coordinates=False,
                                            use_clip_stem=False)


        '''
        Estamos contando el state_joints como un vector de 6 articulaciones.
        '''

        total_dim = 128 + 128 + 6

        self.output_dim = 1024

        self.mlp = MLP(in_features=total_dim,
                       hidden_dims=[1024, 1024],
                       activate_final=True,
                       use_layer_norm=False,
                       dropout_rate=None)

        self.concatenate = Concatenate(model=self.mlp, flatten_time=True)

    def forward(self, batch: dict) -> torch.Tensor:
        '''
        batch = {
            "state_image_1": state_image_1,
            "state_image_2": state_image_2,
            "state_joints": state_joints,
        }
        '''

        image_1 = self.encoder_state_image1(batch["observation.images.laptop"])
        image_2 = self.encoder_state_image2(batch["observation.images.phone"])
        joints = batch["observation.state"]

        x = self.concatenate([image_1, image_2, joints])

        return x
           
class MultiDecoderStates(nn.Module):
    def __init__(self,z_dim: int,mlp_dim: int):
        super(MultiDecoderStates, self).__init__()

        self.mlp_dim = mlp_dim

        self.decoder_state_image1 = ResNet18Decoder(mlp_dim=mlp_dim,
                                                    num_filters=64,
                                                    act=nn.ReLU,
                                                    spatial_coordinates=False)

        self.decoder_state_image2 = ResNet18Decoder(mlp_dim=mlp_dim,
                                                    num_filters=64,
                                                    act=nn.ReLU,
                                                    spatial_coordinates=False)
        
        self.mlp = MLP(in_features=z_dim,
                       hidden_dims=[mlp_dim, mlp_dim],
                       activate_final=True,
                       use_layer_norm=False,
                       dropout_rate=None)
    
    def forward(self, z: torch.Tensor, batch: Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        z = self.mlp(z)
        image_1 = self.decoder_state_image1(z,batch["observation.images.laptop"])
        image_2 = self.decoder_state_image2(z,batch["observation.images.phone"])
        joints = z
        return image_1,image_2,joints #(B, 3, 480, 640], [B, 3, 480, 640]), [B, 1024])
                                                    



if __name__ == '__main__':
    torch.manual_seed(42)
    
    # Definir el dispositivo a usar (GPU si está disponible, CPU en caso contrario)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    B = 8       # Batch size
    z_dim = 16      # Dimensión del espacio latente
    mlp_dim = 1024  # Dimensión interna del decoder

    # Batch dummy para el encoder y para la reconstrucción (se esperan estas formas):
    # - "state_image_1" y "state_image_2": (B, 3, 480, 640)
    # - "state_joints": (B, 6)
    batch = {
        "observation.images.laptop": torch.randn(B, 3, 480, 640).to(device),
        "observation.images.phone": torch.randn(B, 3, 480, 640).to(device),
        "observation.state": torch.randn(B, 6).to(device)
    }

    # Instanciar los módulos dummy.
    dummy_encoder = MultiEncoderStates()
    dummy_decoder = MultiDecoderStates(z_dim=z_dim, mlp_dim=mlp_dim)

    # Crear instancia de BetaVAE.
    beta = 1.0
    weights = [1.0, 1.0, 1.0]  # Peso por cada modalidad

    vae = BetaVAE(dummy_encoder,
                  dummy_decoder,
                  z_dim=z_dim,
                  beta=beta,
                  weights=weights)
    
    vae.model.to(device)  # Mover el modelo completo al dispositivo

    #print(f"Number of parameters: {vae.num_parameters()}")
    # Creamos un optimizador para el modelo.
    optimizer = torch.optim.Adam(vae.model.parameters(), lr=1e-3)

    # --- Test: Forward pass de BetaVaeModel ---
    while True:
        x_hat, mean, logvar = vae.model(batch)
        print("BetaVaeModel Forward:")
        for key, value in x_hat.items():
            print(f"  {key}: {value.shape}")
            print("mean shape:", mean.shape)
            print("logvar shape:", logvar.shape)
        

        # --- Test: train_step ---
        loss, info = vae.train_step(batch, optimizer)
        print("\nTrain Step:")
        print("Loss:", loss)
        print("Info:", info)

        # --- Test: val_step ---
        val_loss, val_info = vae.val_step(batch)
        print("\nValidation Step:")
        print("Val Loss:", val_loss)
        print("Val Info:", val_info)

        # --- Test: predict (encode) ---
        x_hat, mean, logvar = vae.predict(batch)
        print("\nPredict Step (Encoding):")
        print("mean shape:", mean.shape)
        print("logvar shape:", logvar.shape)

        print(x_hat.keys())
        print(x_hat["observation.images.laptop"].shape)
        print(x_hat["observation.images.phone"].shape)
        print(x_hat["observation.state"].shape)

        print("\n¡Test completado con éxito!")