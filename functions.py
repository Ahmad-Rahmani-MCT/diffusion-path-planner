import numpy as np 
from scipy.interpolate import splprep, splev 
from collections import deque 
import matplotlib.pyplot as plt   
from scipy.ndimage import binary_dilation
import torch
from torch.utils.data import Dataset
import os
import random

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"seed set to {seed}")

def generate_single_sample(grid_size: int, num_obstacles: int, min_radius: int, max_radius: int, safety_pixel: int, num_points_path: int): 

    while True:
         
        obs_channel = np.zeros((grid_size, grid_size), dtype=np.float32) 
        start_channel = np.zeros((grid_size, grid_size), dtype=np.float32) 
        goal_channel = np.zeros((grid_size, grid_size), dtype=np.float32) 

        Y, X = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij') # cartesian indexing 

        for _ in range(num_obstacles):
            cx = np.random.randint(0, grid_size)
            cy = np.random.randint(0, grid_size)
            r = np.random.randint(min_radius, max_radius+1)
            dist_sq = (X - cx)**2 + (Y - cy)**2 # square distance from all points in the grid to the center of obstacles
            obs_channel[dist_sq <= r**2] = 1.0 

        # inflate obstacles by 2 pixel (safety margin)
        safe_obs_channel = binary_dilation(obs_channel, iterations=safety_pixel).astype(np.float32)
        free_spaces = np.argwhere(safe_obs_channel == 0) # indices of where is zero

        if len(free_spaces) < 2: # no place for start and goal
            continue 

        start_idx, goal_index = np.random.choice(len(free_spaces), 2, replace=False) 
        start_y, start_x = free_spaces[start_idx] 
        goal_y, goal_x = free_spaces[goal_index] 

        start_channel[start_y, start_x] = 1.0
        goal_channel[goal_y, goal_x] = 1.0 

        # dilating the blobs 
        start_channel = binary_dilation(start_channel, iterations=3).astype(np.float32)
        goal_channel = binary_dilation(goal_channel, iterations=3).astype(np.float32)

        path = find_path_bfs(safe_obs_channel, (start_x, start_y), (goal_x, goal_y))

        if path is not None and len(path) > 3 : 

            path = smooth_path(path, num_points_path)        
            map_tensor = np.stack([obs_channel, start_channel, goal_channel], axis=0)

            return map_tensor, path 
        
def find_path_bfs(grid, start, goal): 
    queue = deque([[start]]) # [[]] because queue refers to a path
    seen = set([start]) 
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)] # each tuple is (dx,dy)
    while queue:
        path = queue.popleft()
        x, y = path[-1] 
        if (x, y) == goal:
            return path 
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]: # because [0] is rows !
                if grid[ny, nx] == 0 and (nx, ny) not in seen: # no obstacle and not already seen ! 
                    queue.append(path + [(nx, ny)]) # branches into separate paths, one for each neighbor. 
                    # BFS explores all possible routes kinda in parallel
                    seen.add((nx, ny)) 
    return None 

def smooth_path(path, num_points): 
    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1] 
    tck, u = splprep([x, y], s=0.5, k=min(3, len(path)-1))  # making the spline
    u_new = np.linspace(0, 1, num_points) # evenly spaced parameters to be evaluated
    x_new, y_new = splev(u_new, tck)
    trajectory = np.column_stack((x_new, y_new))    
    return trajectory

class LoadDataset(Dataset): # inhertance from Dataset class -> directly to DataLoader
    def __init__(self, npz_file_name: str, np_data_map_name: str, np_data_path_name: str, grid_size: int): 

        cwd = os.getcwd() 
        npz_path = os.path.join(cwd, npz_file_name)
        data = np.load(npz_path) 

        # load raw and convert to tensors 
        raw_maps = torch.tensor(data[np_data_map_name], dtype=torch.float32)  
        raw_paths = torch.tensor(data[np_data_path_name], dtype=torch.float32).permute(0,2,1)

        # normalization     
        self.maps = 2 * raw_maps - 1 
        self.paths = 2 * (raw_paths / grid_size) - 1  

    def __len__(self): # needed for inheritance
        return len(self.maps) 
    
    def __getitem__(self, idx): # needed for inheritance
        return self.maps[idx], self.paths[idx]