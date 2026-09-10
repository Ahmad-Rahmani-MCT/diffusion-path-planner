## PARAMETERS FOR DATASET CREATION ## 
NUM_SAMPLES = 100000 # number of samples in the generated dataset
GRID_SIZE = 64 # size of the side of the square map/grid 
NUM_OBSTACLES = 10 # number of obstacles 
MIN_RADIUS = 3 # minimum radius of obstacles 
MAX_RADIUS = 4 # maximum radius of obstacles 
SAFETY_PIXEL = 5 # obstacle dilation iteration
NUM_POINTS_PATH = 64 # number of points in the generated path 
NP_DATA_MAP_NAME = "maps" # name of the map data when saving in .npz
NP_DATA_PATH_NAME = "paths" # name of the path data when saving in .npz
MAP_PATH_DATSET_NAME = "map_path_dataset.npz" # name of saving the dataset

## PARAMETERS FOR THE DIFFUSION MODEL ## 
MAP_EMBEDDING_DIM = 1024 # map encoding dimension 
TIME_EMBEDDING_DIM = 128 # time encoding dimension 

## PARAMETERS FOR THE TRAINING SCRIPT ## 
BATCH_SIZE = 64  
EPOCHS = 300 
LEARNING_RATE = 2e-4 
NUM_DIFFUSION_STEPS = 1000 
TRAIN_PERCENT = 0.7 
VAL_PERCENT = 0.15 
TRAINED_MODEL_WEIGHT_NAME = "diffusion_path_planner_weights.pth"