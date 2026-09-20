import torch
import torch.nn as nn


input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])

# Create a linear layer with 3 input features and 2 output features
linear_layer = nn.Linear(in_features=3, out_features=2)

# Pass the input tensor through the linear layer
# Gewichte bestimmen wie viel Einfluss jeder Eingang auf den Aufgang hat. Bias verschiebt die Ausgabe, um die Vorhersage zu verbessern.

output_tensor = linear_layer(input_tensor)

print("Output Tensor: ", output_tensor)

