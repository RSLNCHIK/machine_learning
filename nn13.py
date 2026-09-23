"""
    Schichten eines Modells einfrieren, um die Gewichte und Biases der 
    Schichten nicht zu aktualisieren. Dies kann nuetzlich sein, wenn man 
    ein vortrainiertes Modell verwenden moechte und nur die letzten Schichten
    des Modells anpassen moechte.
    Wir koennen die Gewichte und Biases der Schichten einfrieren, indem wir
    die Eigenschaft "requires_grad" auf False setzen.
"""

import torch
import torch.nn as nn

import torch.optim as optim

model = nn.Sequential(
    nn.Linear(8, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1)
)

for name, param in model.named_parameters():
    if name == "0.weight":
        # Freeze the weights of the first layer
        param.requires_grad = False

    if name == "0.bias":
        param.requires_grad = False



layer0 = nn.Linear(16, 32)
layer1 = nn.Linear(32, 64)


# Use uniform initialization for the weights of the first layer
nn.init.uniform_(layer0.weight)
nn.init.uniform_(layer1.weight)


model1 = nn.Sequential(
    layer0,
    layer1)

