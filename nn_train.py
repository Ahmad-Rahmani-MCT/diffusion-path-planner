import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, random_split 
from diffusers import DDPMScheduler
from nn_models import DiffusionPlanner 
from functions import LoadDataset, set_seed
from parameters import * 

def train():  

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"compute device: {device}") 

    # data split
    full_dataset = LoadDataset(npz_file_name=MAP_PATH_DATSET_NAME, np_data_map_name= NP_DATA_MAP_NAME, np_data_path_name= NP_DATA_PATH_NAME, grid_size=GRID_SIZE) # DataSet class inheritance
    train_size = int(len(full_dataset)* TRAIN_PERCENT)
    val_size = int(len(full_dataset) * VAL_PERCENT)
    test_size = len(full_dataset) - (train_size + val_size)
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # diffusion model instatiation
    model = DiffusionPlanner(map_embedding_dim=MAP_EMBEDDING_DIM, time_embedding_dim=TIME_EMBEDDING_DIM).to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS) 
    loss_fn = nn.MSELoss() 

    print(f"starting training (train sample size: {train_size}, val sample size: {val_size})") 

    best_val_loss = float('inf')

    for epoch in range(EPOCHS): 
        
        # train
        model.train()
        
        train_loss = 0 
        for maps, clean_path in train_loader:

            maps, clean_path = maps.to(device), clean_path.to(device) 
            
            noise = torch.randn_like(clean_path) 
            bs = clean_path.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long() 
            
            noisy_path = noise_scheduler.add_noise(clean_path, noise, timesteps) 
            
            optimizer.zero_grad() 
            predicted_noise = model(noisy_path, timesteps, maps) 
            loss = loss_fn(predicted_noise, noise) 
            loss.backward() 
            optimizer.step() 
            
            train_loss += loss.item()

        # validation
        model.eval()
        
        val_loss = 0
        with torch.no_grad():
            for maps, clean_path in val_loader:
                maps, clean_path = maps.to(device), clean_path.to(device) 
                
                noise = torch.randn_like(clean_path) 
                bs = clean_path.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long() 
                
                noisy_path = noise_scheduler.add_noise(clean_path, noise, timesteps) 
                predicted_noise = model(noisy_path, timesteps, maps) 
                loss = loss_fn(predicted_noise, noise) 
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # save whenever val improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_" + TRAINED_MODEL_WEIGHT_NAME)

        # stats every 50 epochs
        if (epoch+1) % 10 == 0 : 
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"epoch {epoch+1}/{EPOCHS} | train loss: {avg_train_loss:.5f} | val loss: {avg_val_loss:.5f}") 
    
    print("training complete") 

    # testing
    print("start testing phase") 

    model.load_state_dict(torch.load("best_" + TRAINED_MODEL_WEIGHT_NAME, map_location=device, weights_only=True))

    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        for maps, clean_path in test_loader:
            maps, clean_path = maps.to(device), clean_path.to(device) 
            
            noise = torch.randn_like(clean_path) 
            bs = clean_path.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long() 
            
            noisy_path = noise_scheduler.add_noise(clean_path, noise, timesteps) 
            predicted_noise = model(noisy_path, timesteps, maps) 
            loss = loss_fn(predicted_noise, noise) 
            test_loss += loss.item() 

    avg_test_loss = test_loss / len(test_loader)
    print(f"test loss: {avg_test_loss:.5f}")

if __name__ == "__main__":
    train()