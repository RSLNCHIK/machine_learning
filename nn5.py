import torch
import torch.nn as nn

input_tensor = torch.tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Implement a small neural network with 3 layers using nn.Sequential
model = nn.Sequential(
    nn.Linear(11, 8),
    nn.Linear(8, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1))

output_tensor = model(input_tensor)
print("Output Tensor: ", output_tensor)