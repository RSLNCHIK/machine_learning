"""
    Die ReLU Aktivierungsfunktion ist eine nichtlineare Funktion, 
    die in neuralen Netzwerken verwendet wird.
    Die ReLU Funktion gibt den Eingabewert zurück, wenn er größer als 0 ist,
    und 0, wenn er kleiner oder gleich 0 ist.
    Die ReLU Funktion ist definiert als:
    ReLU(x) = max(0, x)
    Die ReLU Funktion ist eine der am häufigsten verwendeten Aktivierungsfunktionen in neuralen Netzwerken, 
    da sie einfach zu implementieren ist und gute Ergebnisse liefert. 
    (hilft bei der Verschwindung des Gradientenproblems, da
    sie die Gradienten nicht saturieren/saettigen), wie es bei der Sigmoid- oder
    Softmax-Funktion der Fall ist.)
"""


import torch
import torch.nn as nn
import torch.optim as optim

# ReLU activation function using PyTorch
# We can use the nn.ReLU() class to create a ReLU activation function in PyTorch. 
# The nn.ReLU() class is a subclass of the nn.Module class, which means that it can be used as 
# a layer in a neural network model. 
# relu_pytorch = nn.ReLU() is an instance of the nn.ReLU() class,
# which means that it can be used as a layer in a neural network model.
relu_pytorch = nn.ReLU()

x_pos = torch.tensor(2.0)
x_neg = torch.tensor(-3.0)

# Apply the ReLU function to positive and negative inputs

output_pos = relu_pytorch(x_pos)
output_neg = relu_pytorch(x_neg)

"""
    Leaky ReLU Funktion ist eine Variante der ReLU Funktion, die es ermoeglicht,
    dass negative Eingabewerte einen kleinen, nicht-null Ausgabewert haben.
    Die Leaky ReLU Funktion ist definiert als:
    LeakyReLU(x) = x, wenn x > 0
    LeakyReLU(x) = alpha * x, wenn x <= 0
    Die Leaky ReLU Funktion ist eine der am häufigsten verwendeten Aktivierungsfunktionen in neuralen Netzwerken,
    da sie einfach zu implementieren ist und gute Ergebnisse liefert.
    (hilft bei der Verschwindung des Gradientenproblems, da
    sie die Gradienten nicht saturieren/saettigen), wie es bei der Sigmoid- oder
    Softmax-Funktion der Fall ist.)
"""

# Create a Leaky ReLU activation function using PyTorch
# We can use the nn.LeakyReLU() class to create a Leaky ReLU activation function in PyTorch.
# leaky_relupytorch is an instance of the nn.LeakyReLU() class, which means that it can be used as a layer in a neural network model.
leaky_relu_pytorch = nn.LeakyReLU(negative_slope=0.01)

x_input = torch.tensor(-2.0)

# Apply the Leaky ReLU function to the input
output_leaky_relu = leaky_relu_pytorch(x_input)



