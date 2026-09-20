import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

y = [2]

scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# Create a one-hot encoded vector of the label y using PyTorch
one_hot_label = F.one_hot(torch.tensor(y), num_classes=4)


# Create a CrossEntropyLoss object
loss_fn = CrossEntropyLoss()

# Calculate the loss using the one-hot encoded label
# one_hot_label is converted to float to match the expected input type for CrossEntropyLoss 
loss = loss_fn(scores.double(), one_hot_label.double())