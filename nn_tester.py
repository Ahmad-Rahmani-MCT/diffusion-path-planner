from nn_models import * 

batch_size = 4
fake_maps = torch.randn(batch_size, 3, 64, 64)
fake_paths = torch.randn(batch_size, 2, 64) # pytorch wants channels first 
fake_timesteps = torch.randint(0, 100, (batch_size,)) 

model = DiffusionPlanner() 

with torch.no_grad(): 
    output = model.forward(fake_paths, fake_timesteps, fake_maps) 

print(f"map shape {fake_maps.shape}") 
print(f"path shape {fake_paths.shape}") 
print(f"output predicted noise {output.shape}") 