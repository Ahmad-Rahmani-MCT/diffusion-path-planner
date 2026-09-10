import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from nn_models import DiffusionPlanner
from functions import LoadDataset
from parameters import *
import os 
import time

def run_multi_seed_check(test_idx=None, num_samples=12):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionPlanner(map_embedding_dim=MAP_EMBEDDING_DIM, time_embedding_dim=TIME_EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "best_" + TRAINED_MODEL_WEIGHT_NAME),
                                      map_location=device, weights_only=True))
    model.eval()

    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS)
    noise_scheduler.set_timesteps(NUM_DIFFUSION_STEPS)

    dataset = LoadDataset(npz_file_name=MAP_PATH_DATSET_NAME, np_data_map_name=NP_DATA_MAP_NAME,
                           np_data_path_name=NP_DATA_PATH_NAME, grid_size=GRID_SIZE)
    test_idx = test_idx if test_idx is not None else np.random.randint(len(dataset))
    test_map, _ = dataset[test_idx]
    map_input = test_map.unsqueeze(0).to(device)

    with torch.no_grad():
        
        map_features = model.map_encoder(map_input)   # reuse

        all_paths = []
        for seed in range(num_samples):
            torch.manual_seed(seed)
            noisy_path = torch.randn((1, 2, NUM_POINTS_PATH), device=device)
            for t in noise_scheduler.timesteps:
                t_tensor = torch.tensor([t], dtype=torch.long, device=device)
                predicted_noise = model.forward_with_features(noisy_path, t_tensor, map_features)
                noisy_path = noise_scheduler.step(predicted_noise, t, noisy_path).prev_sample
            path = noisy_path.squeeze(0).cpu().numpy()
            all_paths.append((path + 1.0) * (GRID_SIZE / 2))

    test_map_np = test_map.numpy()
    visual_grid = np.stack([test_map_np[0], test_map_np[1], test_map_np[2]], axis=-1)
    visual_grid = (visual_grid * 255).astype(np.uint8)

    plt.figure(figsize=(8, 8))
    plt.imshow(visual_grid)
    for path in all_paths:
        plt.plot(path[0, :], path[1, :], linewidth=1.5, alpha=0.6)
    plt.title(f"{num_samples} denoising runs, same map (idx={test_idx})")

    os.makedirs("generated_paths", exist_ok=True)
    save_path = f"generated_paths/multi_seed_idx{test_idx}_{int(time.time())}.png"
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    run_multi_seed_check()