import torch.nn as nn
import torch.nn.functional as F

class UNetAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super(UNetAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Decoder sin skip connections
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        
        # Activaciones
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.enc1(x))  # (batch, 64, H, W)
        x2 = self.relu(self.enc2(F.max_pool2d(x1, 2)))  # (batch, 128, H/2, W/2)
        latent = self.relu(self.enc3(F.max_pool2d(x2, 2)))  # (batch, 256, H/4, W/4)
        
        # Decoder sin skip connections
        d1 = self.relu(self.dec1(F.interpolate(latent, scale_factor=2, mode='bilinear', align_corners=False)))  # (batch, 128, H/2, W/2)
        d2 = self.relu(self.dec2(F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)))  # (batch, 64, H, W)
        reconstructed = self.sigmoid(self.dec3(d2))  # (batch, input_channels, H, W)
        
        return reconstructed, latent
