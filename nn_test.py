import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from nn_models import DiffusionPlanner
from functions import LoadDataset, set_seed
import os
import time
from parameters import *

def run_inference():

    #set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"compute device: {device}")

    # instatiate model and load weights
    model = DiffusionPlanner(map_embedding_dim=MAP_EMBEDDING_DIM, time_embedding_dim=TIME_EMBEDDING_DIM).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS)
    
    # paths 
    cwd = os.getcwd() 
    weights_path = os.path.join(cwd, ("best_" + TRAINED_MODEL_WEIGHT_NAME))
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print("nn weights loaded successfully")
    except FileNotFoundError:
        print(f"Error: Could not find {weights_path}")
        return

    model.eval()
    dataset = LoadDataset(npz_file_name=MAP_PATH_DATSET_NAME, np_data_map_name=NP_DATA_MAP_NAME, np_data_path_name=NP_DATA_PATH_NAME, grid_size=GRID_SIZE)

    # picking a random map
    test_idx = np.random.randint(len(dataset))
    test_map, real_path = dataset[test_idx]
    map_input = test_map.unsqueeze(0).to(device)

    # noisy path 
    path_shape = (1, 2, NUM_POINTS_PATH)
    noisy_path = torch.randn(path_shape, device=device)
    noise_scheduler.set_timesteps(NUM_DIFFUSION_STEPS) 
    
    # denoising
    print("denoising")
    with torch.no_grad():

        for t in noise_scheduler.timesteps:
        
            t_tensor = torch.tensor([t], dtype=torch.long, device=device)
            predicted_noise = model(noisy_path, t_tensor, map_input)
            noisy_path = noise_scheduler.step(predicted_noise, t, noisy_path).prev_sample

    # unnormalization
    final_path = noisy_path.squeeze(0).cpu().numpy()
    final_path = (final_path + 1.0) * (GRID_SIZE / 2)
    real_path = real_path.numpy()
    real_path = (real_path + 1.0) * (GRID_SIZE / 2) 

    # construction RGB map and plotting
    test_map_np = test_map.numpy()
    visual_grid = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    visual_grid[:, :, 0] = test_map_np[0] * 255 
    visual_grid[:, :, 1] = test_map_np[1] * 255 
    visual_grid[:, :, 2] = test_map_np[2] * 255 

    plt.figure(figsize=(8, 8))
    plt.imshow(visual_grid)
    plt.plot(real_path[0, :], real_path[1, :], color='gray', linestyle='--', linewidth=2, label='Expert Path')
    plt.plot(final_path[0, :], final_path[1, :], color='cyan', linewidth=3, label='DDPM Generated Path')
    plt.title("DDPM Obstacle Avoidance Path Planner")
    plt.legend() 

    os.makedirs("generated_paths", exist_ok=True)
    save_path = f"generated_paths/path_idx{test_idx}_{int(time.time())}.png"
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show() 

if __name__ == "__main__":
    run_inference()