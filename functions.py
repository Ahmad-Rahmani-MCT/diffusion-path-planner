import numpy as np 
from scipy.interpolate import splprep, splev 
from collections import deque 
import matplotlib.pyplot as plt   

def generate_single_sample(grid_size=64, num_obstacles=4, min_radius=3, max_radius=5, num_points_path=50): 
    while True: 
        
        obs_channel = np.zeros((grid_size, grid_size), dtype=np.float32) 
        start_channel = np.zeros((grid_size, grid_size), dtype=np.float32) 
        goal_channel = np.zeros((grid_size, grid_size), dtype=np.float32) 

        # for _ in range(num_obstacles): 
        #     r = np.random.randint(min_radius, max_radius) 
        #     cx, cy = np.random.randint(0, grid_size-r, 2) 
        #     Y, X = np.ogrid[:grid_size, :grid_size] 
        #     dist_from_center = np.sqrt((X-cx)**2 + (Y-cy)**2) 
        #     obs_channel[dist_from_center <= r] 

        Y, X = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        for _ in range(num_obstacles):
            cx = np.random.randint(0, grid_size)
            cy = np.random.randint(0, grid_size)
            r = np.random.randint(min_radius, max_radius)
            
            dist_sq = (X - cx)**2 + (Y - cy)**2
            obs_channel[dist_sq <= r**2] = 1.0

        free_spaces = np.argwhere(obs_channel == 0) 
        if len(free_spaces) < 2: 
            continue 

        start_idx, goal_index = np.random.choice(len(free_spaces), 2, replace=False) 
        start_y, start_x = free_spaces[start_idx] 
        goal_y, goal_x = free_spaces[goal_index] 

        start_channel[start_y, start_x] = 1.0
        goal_channel[goal_y, goal_x] = 1.0

        path = find_path_bfs(obs_channel, (start_x, start_y), (goal_x, goal_y)) 

        if path is not None and len(path) > 3 : 
            
            trajectory = smooth_path(path, num_points_path)         

            map_tensor = np.stack([obs_channel, start_channel, goal_channel], axis=0) 
            
            return map_tensor, trajectory 
        
def find_path_bfs(grid, start, goal): 
    
    queue = deque([[start]]) 
    seen = set([start]) 
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)] 

    while queue:
        
        path = queue.popleft()
        x, y = path[-1] 

        if (x, y) == goal:
            return path 
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check bounds and if obstacle
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                if grid[ny, nx] == 0 and (nx, ny) not in seen:
                    queue.append(path + [(nx, ny)])
                    seen.add((nx, ny)) 

    return None 

def smooth_path(path, num_points=50): 

    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1] 

    tck, u = splprep([x, y], s=5.0, k=min(3, len(path)-1)) 

    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    trajectory = np.column_stack((x_new, y_new))
    return trajectory