import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder3D(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(Encoder3D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)  # Downsample by 2
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3D, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),  # Upsample by 2
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.decoder(x)

class TimeDecoder(nn.Module):
    def __init__(self, input_dim):
        super(TimeDecoder, self).__init__()
        # Intermediate dimension
        intermediate_dim = input_dim * 2  # You can adjust this multiplier as necessary
        
        # Define a series of Linear layers and ReLUs
        self.layers = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim  , intermediate_dim * 2 ),
            nn.ReLU(),
            nn.Linear(intermediate_dim * 2, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.ReLU(),
            nn.Linear(intermediate_dim // 2, 2)  # Outputs the shape and scale parameters
        )
        self.softplus = nn.Softplus() 

    def forward(self, z):
        return self.softplus(self.layers(z))
