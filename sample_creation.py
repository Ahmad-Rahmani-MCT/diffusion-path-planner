import numpy as np
from functions import generate_single_sample 
from parameters import *

print(f"generating {NUM_SAMPLES} dataset samples")

all_maps = []
all_paths = []

for i in range(NUM_SAMPLES):
    if i % 100 == 0:
        print(f"Generated {i} / {NUM_SAMPLES}")
        
    m, t = generate_single_sample(grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES, min_radius=MIN_RADIUS, max_radius=MAX_RADIUS, safety_pixel=SAFETY_PIXEL, num_points_path=NUM_POINTS_PATH)
    all_maps.append(m)
    all_paths.append(t)
    
maps_np = np.array(all_maps, dtype=np.float32)
paths_np = np.array(all_paths, dtype=np.float32)

np.savez_compressed('map_path_dataset.npz', **{NP_DATA_MAP_NAME: maps_np, NP_DATA_PATH_NAME: paths_np})

print("dataset saved to 'map_path_dataset.npz'")
print(f"maps shape: {maps_np.shape}")
print(f"trajectories shape: {paths_np.shape}")