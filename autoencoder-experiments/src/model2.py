import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super(UNetAutoencoder, self).__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_layer = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.enc1(x)  # (batch, 64, H, W)
        x2 = self.enc2(F.max_pool2d(x1, 2))  # (batch, 128, H/2, W/2)
        x3 = self.enc3(F.max_pool2d(x2, 2))  # (batch, 256, H/4, W/4)

        latent = self.bottleneck(F.max_pool2d(x3, 2))  # (batch, 1024, H/8, W/8)

        d1 = self.dec1(F.interpolate(latent, scale_factor=2, mode='nearest')) + x3
        d2 = self.dec2(F.interpolate(d1, scale_factor=2, mode='nearest'))

        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + x2

        d3 = self.dec3(F.interpolate(d2, scale_factor=2, mode='nearest'))

        if d3.shape[2:] != x1.shape[2:]:
            d3 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + x1

        reconstructed = self.sigmoid(self.final_layer(d3))

        return reconstructed, latent
