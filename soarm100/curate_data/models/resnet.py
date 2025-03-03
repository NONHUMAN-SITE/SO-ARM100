import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Helper Modules
# ---------------------------------------------------------------------

class SpatialCoordinates(nn.Module):
    """
    Añade dos canales extra que contienen las coordenadas espaciales
    linealmente espaciadas entre -1 y 1 (para el eje x y para el eje y).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x tiene forma (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)
        coords = torch.stack([xs, ys], dim=0)  # (2, H, W)
        coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)
        return torch.cat([x, coords], dim=1)

class ResNetBlock(nn.Module):
    """
    Bloque básico de ResNet.
    Se compone de dos convoluciones 3x3 con GroupNorm (32 grupos) seguidas de ReLU.
    Si las dimensiones de la entrada y de la salida son distintas, se proyecta la
    rama residual mediante una convolución 1x1 (o 3x3 en el caso de upsampling).
    """
    def __init__(self, in_channels, out_channels, stride=1, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Primera convolución (en la rama principal)
        conv_stride = stride if not transpose else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=conv_stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, out_channels, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

        # Segunda convolución
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, out_channels, eps=1e-5)
        # Inicializar los parámetros de gn2 a cero (similar a scale_init en JAX)
        nn.init.constant_(self.gn2.weight, 0.0)

        # Proyección para la rama residual en caso de desajuste de dimensiones:
        self.need_proj = (in_channels != out_channels) or (stride != 1)
        if self.need_proj:
            if transpose:
                # Si se usa "transpose", se hace upsampling usando interpolate y se emplea una convolución 3x3.
                self.conv_proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                self.conv_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.gn_proj = nn.GroupNorm(32, out_channels, eps=1e-5)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        if self.transpose and self.stride > 1:
            # Upsample usando nearest neighbor
            out = F.interpolate(out, scale_factor=self.stride, mode="nearest")
        out = self.conv2(out)
        out = self.gn2(out)

        if self.need_proj:
            if self.transpose and self.stride > 1:
                residual = F.interpolate(residual, scale_factor=self.stride, mode="nearest")
            residual = self.conv_proj(residual)
            residual = self.gn_proj(residual)
        out = self.relu(residual + out)
        return out

class SpatialSoftmax(nn.Module):
    """
    Calcula los keypoints espaciales mediante softmax espacial.
    Si se proporciona num_kp, se aplica una convolución 1x1 para proyectar al número deseado.
    El resultado son las coordenadas esperadas por canal (concatenando las coord. x e y).
    """
    def __init__(self, num_kp, temperature=1.0):
        super().__init__()
        self.num_kp = num_kp
        self.temperature = temperature
        # Utilizamos una capa lazy para definir la cantidad de canales de entrada en el primer forward.
        self.keypoints_conv = nn.LazyConv2d(self.num_kp, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        # x tiene forma (B, C, H, W)
        x = self.keypoints_conv(x)  # ahora x tiene shape (B, num_kp, H, W)
        B, C, H, W = x.shape

        # Crear las coordenadas espaciales (linealmente entre -1 y 1)
        pos_x = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        pos_y = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        # Uso meshgrid para obtener las grillas
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')  # ambos de shape (H, W)
        # Utiliza .reshape en lugar de .view para evitar problemas con tensores no contiguos.
        grid_x = grid_x.reshape(1, 1, H * W)  # (1, 1, H*W)
        grid_y = grid_y.reshape(1, 1, H * W)

        # Aplanar el mapa espacial
        x_flat = x.reshape(B, C, H * W)
        softmax_attention = F.softmax(x_flat / self.temperature, dim=2)  # sobre la dimensión espacial

        expected_x = torch.sum(softmax_attention * grid_x, dim=2)  # (B, C)
        expected_y = torch.sum(softmax_attention * grid_y, dim=2)  # (B, C)
        # Concatenar las coordenadas esperadas
        expected = torch.cat([expected_x, expected_y], dim=1)  # (B, 2 * C)

        return expected

class AttentionPool2d(nn.Module):
    """
    Implementa el Attention Pooling similar al de CLIP.
    Se aplana la entrada espacialmente, se le agrega un token de query (el promedio)
    y se suma un embedding posicional aprendido.
    """
    def __init__(self, input_dim, num_heads, spatial_size):
        """
        spatial_size: número total de posiciones (H*W) esperado.
        """
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.spatial_size = spatial_size
        # Se crea un embedding posicional para (H*W + 1) tokens.
        self.pos_emb = nn.Parameter(torch.randn(1, spatial_size + 1, input_dim) * (1.0 / math.sqrt(input_dim)))
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def forward(self, x):
        # x tiene forma (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        global_token = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        x = torch.cat([global_token, x], dim=1)  # (B, H*W+1, C)
        x = x + self.pos_emb  # se suma el embedding posicional
        # nn.MultiheadAttention requiere (L, B, C)
        x = x.transpose(0, 1)
        # Se usa el token de query (el primero) para la atención
        attn_output, _ = self.attn(query=x[0:1], key=x, value=x)
        return attn_output.squeeze(0)  # (B, C)

# ---------------------------------------------------------------------
# Clases principales de ResNet
# ---------------------------------------------------------------------

class ResNet(nn.Module):
    """
    Implementa ResNetV1.5 (con group norm en lugar de BatchNorm), con la opción
    de sumar coordenadas espaciales y con ramas de pooling especiales.
    """
    def __init__(
        self,
        in_channels=3,
        stage_sizes=(3, 4, 6, 3),
        num_filters=64,
        num_kp=None,
        spatial_coordinates=False,
        attention_pool=False,
        average_pool=False,
        use_clip_stem=False,
    ):
        super().__init__()
        self.use_clip_stem = use_clip_stem
        self.spatial_coordinates = spatial_coordinates

        if use_clip_stem:
            # Versión CLIP del stem
            self.conv_stem_1 = nn.Conv2d(in_channels, num_filters // 2, kernel_size=3, stride=2, padding=1, bias=False)
            self.gn_stem_1 = nn.GroupNorm(32, num_filters // 2, eps=1e-5)
            self.conv_stem_2 = nn.Conv2d(num_filters // 2, num_filters // 2, kernel_size=3, stride=1, padding=1, bias=False)
            self.gn_stem_2 = nn.GroupNorm(32, num_filters // 2, eps=1e-5)
            self.conv_stem_3 = nn.Conv2d(num_filters // 2, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
            self.gn_stem_3 = nn.GroupNorm(32, num_filters, eps=1e-5)
            self.avgpool_clip = nn.AvgPool2d(kernel_size=2, stride=2)
            current_channels = num_filters
        else:
            # Stem estándar
            self.conv_init = nn.Conv2d(in_channels, num_filters, kernel_size=7, stride=2, padding=3, bias=False)
            self.gn_init = nn.GroupNorm(32, num_filters, eps=1e-5)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            current_channels = num_filters

        if spatial_coordinates:
            self.spatial_coords = SpatialCoordinates()

        # Construir las etapas principales de la red  
        self.stages = nn.ModuleList()
        for i, num_blocks in enumerate(stage_sizes):
            stage = []
            for j in range(num_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                out_channels = num_filters * (2 ** i)
                block = ResNetBlock(in_channels=current_channels, out_channels=out_channels, stride=stride, transpose=False)
                stage.append(block)
                current_channels = out_channels
            self.stages.append(nn.Sequential(*stage))

        # Capa de pooling final:
        self.num_kp = num_kp
        self.attention_pool = attention_pool
        self.average_pool = average_pool
        if num_kp is not None:
            self.spatial_softmax = SpatialSoftmax(num_kp=num_kp, temperature=1.0)
        elif attention_pool:
            # Para AttentionPool2d se debe conocer el número de tokens espaciales. 
            # Suponemos que la resolución es fija (depende del tamaño de la imagen y el stem).
            # Aquí se establece un valor de ejemplo; en una implementación real se deberá ajustar.
            spatial_size = 7 * 7  # ejemplo
            self.attention_pool_layer = AttentionPool2d(input_dim=current_channels, num_heads=current_channels // 16, spatial_size=spatial_size)

    def forward(self, x, goal=None, train=True):
        # x se asume de forma (B, C, H, W) con valores en [0, 1]
        if goal is not None:
            # Si se proporciona un objetivo, se concatena a lo largo del canal
            x = torch.cat([x, goal.expand_as(x)], dim=1)
        # Reescalar de [0,1] a [-1,1]
        x = 2 * x - 1

        if self.spatial_coordinates:
            x = self.spatial_coords(x)

        if self.use_clip_stem:
            x = self.conv_stem_1(x)
            x = self.gn_stem_1(x)
            x = F.relu(x)
            x = self.conv_stem_2(x)
            x = self.gn_stem_2(x)
            x = F.relu(x)
            x = self.conv_stem_3(x)
            x = self.gn_stem_3(x)
            x = F.relu(x)
            x = self.avgpool_clip(x)
        else:
            x = self.conv_init(x)
            x = self.gn_init(x)
            x = F.relu(x)
            x = self.maxpool(x)

        for stage in self.stages:
            x = stage(x)

        if self.num_kp is not None:
            x = self.spatial_softmax(x)
        elif self.attention_pool:
            x = self.attention_pool_layer(x)
        elif self.average_pool:
            x = x.mean(dim=(2, 3))
        return x

# ---------------------------------------------------------------------
# ResNet18: estructura con 2 bloques por etapa
# ---------------------------------------------------------------------
class ResNet18(ResNet):
    def __init__(self, in_channels=3, num_filters=64, num_kp=64, **kwargs):
        # Las etapas de ResNet18 son (2, 2, 2, 2)
        stage_sizes = (2, 2, 2, 2)
        super().__init__(
            in_channels=in_channels,
            stage_sizes=stage_sizes,
            num_filters=num_filters,
            num_kp=num_kp,
            **kwargs,
        )

# ---------------------------------------------------------------------
# Pequeño test case
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Creamos un modelo ResNet18 con num_kp=64
    model = ResNet18(in_channels=3,
                     num_kp=64,
                     spatial_coordinates=False,
                     use_clip_stem=False)
    model.eval()  # Modo evaluación

    # Creamos un tensor aleatorio (valores en [0,1]) con forma (B, C, H, W)
    x = torch.rand(2,3, 480, 640)

    # Hacemos un forward
    with torch.no_grad():
        output = model(x)
    # Como num_kp=64, esperamos que la salida tenga 2*64=128 características por batch.
    print("Output shape:", output.shape)
