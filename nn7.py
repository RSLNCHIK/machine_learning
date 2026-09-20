# One-Hot-Coding implementation in Python
import torch
import torch.nn.functional as F
import numpy as np

y = 1
num_classes = 3

# Create a one-hot encoded vector using Numpy
# Manaully create a one-hot encoded vector
one_hot_numpy = np.array([0, 1, 0])

one_hot_pytorch = F.one_hot(torch.tensor(y), num_classes=num_classes)

print("One-Hot Encoded Vector (Numpy): ", one_hot_numpy)
print("One-Hot Encoded Vector (PyTorch): ", one_hot_pytorch)

