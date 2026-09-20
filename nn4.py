import torch
import torch.nn as nn

input_tensor = torch.tensor([[3, 4, 6, 2, 3, 6, 8, 9]])

# Implement a small neural network using nn.Sequential
model = nn.Sequential(
    nn.Linear(8, 5),
    nn.Sigmoid()
    )

# Pass the input tensor through the model
output_tensor = model(input_tensor)

print("Output Tensor: ", output_tensor)