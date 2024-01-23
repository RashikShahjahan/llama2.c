import torch

# Replace 'your_file.pt' with the path to your .pt file
file_path = '/home/rashik/ckpt.pt'

# Load the file
data = torch.load(file_path, map_location=torch.device('cpu'))

# Print the type and contents of the loaded data
print(type(data))
print(data)
