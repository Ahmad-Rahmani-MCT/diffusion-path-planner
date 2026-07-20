## PARAMETERS FOR DATASET CREATION ## 
NUM_SAMPLES = 100 # number of samples in the generated dataset
GRID_SIZE = 64 # size of the side of the square map/grid 
NUM_OBSTACLES = 10 # number of obstacles 
MIN_RADIUS = 3 # minimum radius of obstacles 
MAX_RADIUS = 4 # maximum radius of obstacles 
SAFETY_PIXEL = 2 # obstacle dilation iteration
NUM_POINTS_PATH = 64 # number of points in the generated path 
NP_DATA_MAP_NAME = "maps" # name of the map data when saving in .npz
NP_DATA_PATH_NAME = "paths" # name of the path data when saving in .npz
MAP_PATH_DATSET_NAME = "map_path_dataset.npz" # name of saving the dataset

## PARAMETERS FOR THE DIFFUSION MODEL ## 
MAP_EMBEDDING_DIM = 16 # map encoding dimension 
TIME_EMBEDDING_DIM = 64 # time encoding dimension 