import torch
import torch.nn as nn

# Define the input tensor with 8 features
input_tensor = torch.tensor([[2, 3, 6, 7, 9, 3, 2, 1]], dtype=torch.float32)

# Create a linear layer with 8 input features and 4 output features
linear_layer = nn.Linear(in_features=8, out_features=4)

model = nn.Sequential(
    nn.Linear(8, 4),
    nn.Linear(4, 1)
    )

output_tensor = model(input_tensor)

print("Output Tensor: ", output_tensor)

