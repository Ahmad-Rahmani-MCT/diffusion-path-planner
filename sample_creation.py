import numpy as np
from functions import generate_single_sample, set_seed
from parameters import *

set_seed(42)

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

np.savez_compressed(MAP_PATH_DATSET_NAME, **{NP_DATA_MAP_NAME: maps_np, NP_DATA_PATH_NAME: paths_np})

print(f"dataset saved to '{MAP_PATH_DATSET_NAME}'")
print(f"maps shape: {maps_np.shape}")
print(f"paths shape: {paths_np.shape}")