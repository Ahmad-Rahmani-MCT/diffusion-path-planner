import torch
import torch.nn as nn
from diffusers import UNet1DModel 

class MapEncoder(nn.Module): 

    def __init__(self, embedding_dim=128):
        super().__init__() 

        self.cnn = nn.Sequential(
            # input [batch, 3, 64, 64] 
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # output [batch, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(), 

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # output: [Batch, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # output: [Batch, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(), # flatten 3d to 1d

            # 128*8*8
            nn.Linear(128*8*8, embedding_dim), # output [batch, embedding_dim]
            nn.ReLU()
            )
    
    def forward(self, x): 
        return self.cnn(x)
    
class DiffusionPlanner(nn.Module):

    def __init__(self, condition_dim=128, sample_size=64, in_channels=2, out_channels=2):
        super().__init__()

        self.map_encoder = MapEncoder(embedding_dim=condition_dim) 

        self.unet = UNet1DModel(
            sample_size= sample_size, # path length 
            in_channels= in_channels + condition_dim, # x and y
            out_channels= out_channels, # x and y
            down_block_types=("DownBlock1D", "AttnDownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "AttnUpBlock1D", "UpBlock1D"), 
            block_out_channels=(64, 128, 256),
        )   

    def forward(self, noisy_path, timestep, map_image): 
        
        # noisy_trajectory: shape [Batch, 2, 50]
        # timestep: shape [Batch]
        # map_image: shape [Batch, 3, 64, 64] 

        map_features = self.map_encoder(map_image) # summarize map to [batch, 128]  

        map_features_expanded = map_features.unsqueeze(-1).repeat(1, 1, noisy_path.shape[-1]) # [batch, 128,50]

        model_input = torch.cat([noisy_path, map_features_expanded], dim=1)

        noise_pred = self.unet(
            sample=model_input,
            timestep=timestep,
        ).sample  

        return noise_pred