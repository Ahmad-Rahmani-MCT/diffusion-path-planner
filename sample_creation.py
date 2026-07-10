from functions import *

NUM_SAMPLES = 1000 
GRID_SIZE = 64 
NUM_OBSTACLES = 10 
MIN_RADIUS = 3 
MAX_RADIUS = 4 
NUM_POINTS_PATH = 80 


print(f"generating {NUM_SAMPLES} dataset samples")

all_maps = []
all_trajectories = []

for i in range(NUM_SAMPLES):
    if i % 100 == 0:
        print(f"Generated {i} / {NUM_SAMPLES}")
        
    m, t = generate_single_sample(grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES, min_radius=MIN_RADIUS, max_radius=MAX_RADIUS, num_points_path=NUM_POINTS_PATH)
    all_maps.append(m)
    all_trajectories.append(t)
    
maps_np = np.array(all_maps, dtype=np.float32)
traj_np = np.array(all_trajectories, dtype=np.float32)

np.savez_compressed('map_path_dataset.npz', maps=maps_np, trajectories=traj_np)

print("dataset saved to 'map_path_dataset.npz'")
print(f"maps shape: {maps_np.shape}")
print(f"trajectories shape: {traj_np.shape}")