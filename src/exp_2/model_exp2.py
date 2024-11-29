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
    def __init__(self, input_dim, num_times):
        super(TimeDecoder, self).__init__()
        # Intermediate dimensions and time-aware layers
        intermediate_dim = input_dim * 2  # You can adjust this multiplier
        self.layers = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim * 2),
            nn.ReLU(),
            nn.Linear(intermediate_dim * 2, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.ReLU(),
        )
        
        # Separate output layers
        self.surv = nn.Linear(intermediate_dim // 2, num_times)  # Outputs survival probabilities for all time points
        self.haz = nn.Linear(intermediate_dim // 2, num_times)  # Outputs hazards for all time points
        
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.layers(z)
        
        # Survival and hazard for all times
        surv = torch.sigmoid(self.surv(x))  # Sigmoid to ensure values between 0 and 1
        haz = self.relu(self.haz(x))       # ReLU for non-negative hazard rates
        
        return surv, haz

