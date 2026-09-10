import torch
import torch.nn as nn
import torch.nn.functional as F

class MapEncoder(nn.Module):

    def __init__(self, map_embedding_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*8*8, map_embedding_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.cnn(x)

class Conv1DBlock(nn.Module):

    """
    1D convolutional block with conditioning.
    """
    
    def __init__(self, in_channels, out_channels, time_embedding_dim, map_embedding_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

        # film 
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)

        self.film_mlp = nn.Sequential(
        nn.Linear(map_embedding_dim, out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, 2 * out_channels)
        )

        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias) 

    def forward(self, x, time_embedding, map_embedding):
        
        h = self.relu(self.conv1(x))
        t_cond = self.time_mlp(time_embedding).unsqueeze(-1)
        h = h + t_cond
        film_params = self.film_mlp(map_embedding)
        gamma, beta = film_params.chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(-1)) * h + beta.unsqueeze(-1)

        # second convolution
        h = self.relu(self.conv2(h))

        return h

class CustomConditionalUNet(nn.Module):
    def __init__(self, map_embedding_dim, time_embedding_dim):
        super().__init__()

        # time embedding layer
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        # down path (changing feature sizes)
        self.down1 = Conv1DBlock(2, 128, time_embedding_dim, map_embedding_dim)
        self.down2 = Conv1DBlock(128, 256, time_embedding_dim, map_embedding_dim)
        self.down3 = Conv1DBlock(256, 512, time_embedding_dim, map_embedding_dim)

        # upward path
        self.up1 = Conv1DBlock(512 + 256, 256, time_embedding_dim, map_embedding_dim)
        self.up2 = Conv1DBlock(256 + 128, 128, time_embedding_dim, map_embedding_dim)

        # Final prediction layer
        self.final_conv = nn.Conv1d(128, 2, kernel_size=1)

    def forward(self, x, timestep, map_features):
        # process time
        t = timestep.float().unsqueeze(-1) # (B, 1)
        t_emb = self.time_embed(t) # (B, 64)

        # down
        d1 = self.down1(x, t_emb, map_features)
        d1_pool = F.avg_pool1d(d1, kernel_size=2)

        d2 = self.down2(d1_pool, t_emb, map_features)
        d2_pool = F.avg_pool1d(d2, kernel_size=2)

        mid = self.down3(d2_pool, t_emb, map_features)

        # up
        u1 = F.interpolate(mid, scale_factor=2, mode='nearest')
        u1 = torch.cat([u1, d2], dim=1) # skip connection from encoder
        u1 = self.up1(u1, t_emb, map_features)

        u2 = F.interpolate(u1, scale_factor=2, mode='nearest')
        u2 = torch.cat([u2, d1], dim=1) # skip connection from encoder
        u2 = self.up2(u2, t_emb, map_features)

        # output
        return self.final_conv(u2)

class DiffusionPlanner(nn.Module):
    def __init__(self, map_embedding_dim, time_embedding_dim):
        super().__init__()

        self.map_encoder = MapEncoder(map_embedding_dim=map_embedding_dim)
        self.unet = CustomConditionalUNet(map_embedding_dim=map_embedding_dim, time_embedding_dim=time_embedding_dim)

    def forward(self, noisy_path, timestep, map_image):

        map_features = self.map_encoder(map_image)
        noise_pred = self.unet(noisy_path, timestep, map_features)
        return noise_pred

    def forward_with_features(self, noisy_path, timestep, map_features):
        return self.unet(noisy_path, timestep, map_features)