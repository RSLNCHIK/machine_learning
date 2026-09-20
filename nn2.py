import torch
import torch.nn as nn


model = nn.Sequential(
    nn.Linear(9, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1)
    )

total = 0

# Calculate the total number of parameters in the model
# model.parameters() returns an iterator over all the parameters in the model, and param.numel() returns the number of elements in each parameter tensor. By summing up the number of elements for all parameters, we can get the total number of parameters in the model.
# The parameters include weights and biases for each layer. The total number of parameters is important for understanding the model's capacity and complexity.
for param in model.parameters():
    total += param.numel()

print("Total number of parameters in the model: ", total)
