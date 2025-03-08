from typing import Optional, Callable, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    MLP con capas ocultas definidas en hidden_dims.
    
    Se utiliza una función de activación en cada capa (incluso en la última si activate_final==True),
    puede aplicar LayerNorm y dropout opcionalmente.
    
    Por defecto se usa:
       hidden_dims = [1024, 1024]
       activate_final = True
    """
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [1024, 1024],
        activation: Callable[[], nn.Module] = nn.ReLU,
        activate_final: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate

        layers = []
        current_dim = in_features

        for i, dim in enumerate(hidden_dims):
            # Capa lineal con inicialización Xavier
            linear = nn.Linear(current_dim, dim)
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)
            layers.append(linear)

            if self.use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            # Aplica dropout en las capas intermedias si se ha especificado
            if (i < len(hidden_dims) - 1) and (dropout_rate is not None and dropout_rate > 0):
                layers.append(nn.Dropout(p=dropout_rate))
            # Se aplica la función de activación:
            # Se activa en todas las capas excepto que sea la última y activate_final==False
            if (i < len(hidden_dims) - 1) or self.activate_final:
                layers.append(activation())
            current_dim = dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Concatenate(nn.Module):
    """
    Concatenador que agrupa distintas modalidades en un único vector.
    
    Si flatten_time es True, se aplana cada tensor (excepto la dimensión batch)
    y luego se concatenan a lo largo de la última dimensión.
    
    Si se proporciona un modelo adicional (por ejemplo, un MLP), se le aplica
    a la salida concatenada.
    
    Ejemplo de uso: se esperan dos imágenes y un vector de acciones.
      - Las imágenes procesadas por ResNet18 resultan en tensores de [B, 128] cada uno.
      - Por ejemplo, el tensor de acciones viene en [B, 10, 6] y se aplana a [B, 60].
      - La concatenación final será de forma [B, 128+128+60] = [B, 316].
    """
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        flatten_time: bool = True,
    ):
        super().__init__()
        self.model = model
        self.flatten_time = flatten_time

    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        # Ordena por clave para asegurar consistencia
        if self.flatten_time:
            # Aplana cada tensor excepto la dimensión batch: (B, *) --> (B, -1)
            x = torch.cat([m.reshape(m.shape[0], -1) for m in modalities], dim=-1)
        else:
            # Alternativamente, si se requiere conservar la dimensión temporal (B, T, *).
            x = torch.cat([m.reshape(m.shape[0], m.shape[1], -1) for m in modalities], dim=-1)
        
        if self.model is not None:
            x = self.model(x)
        return x

# ---------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Simulamos un batch de tamaño 2
    batch_size = 2

    # Creamos tensores aleatorios que simulan:
    # - Dos imágenes procesadas por ResNet18: [B, 128] cada una
    # - Un vector de acciones: [B, 10, 6]
    modalities = {
        'image1': torch.randn(batch_size, 128),  # Primera imagen procesada
        'image2': torch.randn(batch_size, 128),  # Segunda imagen procesada
        'actions': torch.randn(batch_size, 10, 6)  # Vector de acciones
    }

    # Calculamos la dimensión de entrada para el MLP
    # (128 + 128 + 10*6 = 316)
    total_dim = 128 + 128 + 10 * 6

    # Creamos el MLP con las dimensiones especificadas
    mlp = MLP(
        in_features=total_dim,
        hidden_dims=[1024, 1024],
        activate_final=True,
        use_layer_norm=False,
        dropout_rate=None
    )

    # Creamos el concatenador con el MLP
    concatenator = Concatenate(model=mlp, flatten_time=True)

    # Procesamos los datos
    output = concatenator(modalities)

    # Imprimimos las formas de los tensores en cada paso
    print("\nFormas de los tensores de entrada:")
    for k, v in modalities.items():
        print(f"{k}: {v.shape}")

    print("\nForma del tensor de salida:")
    print(f"output: {output.shape}")  # Debería ser [2, 1024] (batch_size, última capa del MLP)

    # Verificamos que las dimensiones son correctas
    assert output.shape == (batch_size, 1024), "¡La forma del tensor de salida no es la esperada!"
    print("\n¡Test completado con éxito!")
