import numpy as np 
import matplotlib.pyplot as plt 
from parameters import *

map_path_dataset = np.load(MAP_PATH_DATSET_NAME) 
map_dataset = map_path_dataset[NP_DATA_MAP_NAME] 
path_dataset = map_path_dataset[NP_DATA_PATH_NAME]

print(f"maps dataset shape {map_dataset.shape}") 
print(f"paths dataset shape {path_dataset.shape}")

sample_index = np.random.randint(0, len(map_dataset)) 
sample_map = map_dataset[sample_index,:,:,:]
sample_path = path_dataset[sample_index, :,:] 

print(f"sample map dataset shape {sample_map.shape}") 
print(f"sample path dataset shape {sample_path.shape}")
print(f"random sample obstacles {np.sum(sample_map[0]==1.0)} out of pixels {np.size(sample_map[0])}") 

visual_grid = np.zeros((np.size(sample_map, axis=1), np.size(sample_map, axis=2), 3)) 
visual_grid[:, :, 0] = sample_map[0, :, :] * 255 # obstacle red
visual_grid[:, :, 1] = sample_map[1, :, :] * 255 # start green
visual_grid[:, :, 2] = sample_map[2, :, :] * 255 # goal blue

plt.figure(figsize=(6,6))
plt.imshow(visual_grid)
plt.plot(sample_path[:, 0], sample_path[:, 1], color='cyan', linewidth=2, label='smoothed BFS path')
plt.title("generated sample dataset")
plt.legend()
plt.show()