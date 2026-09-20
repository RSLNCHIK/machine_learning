import torch
import torch.nn as nn

input_tensor = torch.tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Update the neural network to have 4 layers using nn.Sequential
model = nn.Sequential(
    nn.Linear(11, 20),
    nn.Linear(20, 12),
    nn.Linear(12, 6),
    nn.Linear(6, 4),
    nn.Softmax(dim=-1)
    )


output_tensor = model(input_tensor)
print("Output Tensor: ", output_tensor)