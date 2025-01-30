import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ Bloque residual mejorado con proyección para manejar cambio de dimensiones """
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        
        # 🔹 Convoluciones principales
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 🔥 Proyección si hay cambio de dimensiones
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # 🔥 Si los canales ya coinciden, no hacemos nada

    def forward(self, x):
        residual = self.shortcut(x)  # 🔹 Transformamos x si es necesario
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # 🔹 Ahora los tamaños siempre coinciden
        return F.relu(x)

class UNetAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super(UNetAutoencoder, self).__init__()

        # 📌 Codificador con bloques residuales
        self.enc1 = ResidualBlock(input_channels, 64)  # (batch, 64, H, W)
        self.enc2 = ResidualBlock(64, 128, downsample=True)  # (batch, 128, H/2, W/2)
        self.enc3 = ResidualBlock(128, 256, downsample=True)  # (batch, 256, H/4, W/4)
        self.enc4 = ResidualBlock(256, 512, downsample=True)  # (batch, 512, H/8, W/8)

        # 🔥 Embedding del espacio latente (proyección compacta)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),  # Reducción de dimensionalidad
            nn.LayerNorm([256, 56, 28]),  # Normalización en el espacio latente
            nn.ReLU(inplace=True)
        )

        # 📌 Decodificador con bloques residuales
        self.dec1 = ResidualBlock(256, 512)  # (batch, 512, H/8, W/8)
        self.dec2 = ResidualBlock(512, 256)  # (batch, 256, H/4, W/4)
        self.dec3 = ResidualBlock(256, 128)  # (batch, 128, H/2, W/2)
        self.dec4 = ResidualBlock(128, 64)   # (batch, 64, H, W)

        # 🚀 Capa final con atención espacial
        self.attention = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.final_layer = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.enc1(x)  
        x2 = self.enc2(F.max_pool2d(x1, 2))  
        x3 = self.enc3(F.max_pool2d(x2, 2))  
        x4 = self.enc4(F.max_pool2d(x3, 2))  

        # 🔹 Ajustar tamaño de `x4` antes de la bottleneck para evitar errores
        x4 = F.interpolate(x4, size=(56, 28), mode='bilinear', align_corners=False)

        latent = self.bottleneck(x4)  

        # 🔹 Ajustar `latent` antes de sumarlo con `x3`
        d1 = self.dec1(F.interpolate(latent, size=x3.shape[2:], mode='bilinear', align_corners=False)) + x3
        d2 = self.dec2(F.interpolate(d1, size=x2.shape[2:], mode='bilinear', align_corners=False))

        d2 = d2 + x2

        d3 = self.dec3(F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False))

        d3 = d3 + x1

        reconstructed = self.sigmoid(self.final_layer(d3))

        return reconstructed, latent


