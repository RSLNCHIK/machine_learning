"""
    Hier uebe ich den Zugriff auf die verschiedensten Schichten eines PyTorch Modells
    Dafuer wird ein einfaches Modell erstellt, das aus lineren Schichten besteht.
    Ich werde dann das Modell indizieren, um auf die verschiedenen Schichten zuzugreifen und
    die Gewichte und Biases (Verzerrungen) der Schichten zu aktualisieren.
    Auf die Modellparameter zugreifen!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

model = nn.Sequential(
    nn.Linear(16, 8),
    nn.Linear(8, 2),
    nn.Linear(2, 1)
    )

# Acsess the weight of the first layer of the model
weight_0 = model[0].weight
print("Weight of the first layer: ", weight_0)

# Access the bias of the second layer of the model
bias_1= model[1].bias

print("Bias of the second layer: ", bias_1)

"""
    Gewichte und Biases aktualisieren

"""
weight0 = model[0].weight
weight1 = model[1].weight
weight2 = model[2].weight

# Gradients for the weights an biases
# How we make that? We can use the backward() method to compute the gradients of the weights and biases with respect to a loss function.

# grad is a tensor that contains the gradients of the weights and biases with respect to the loss function. We can access the gradients of the weights and biases by calling the .grad attribute of the weight and bias tensors.
grads0 = weight0.grad
grads1 = weight1.grad
grads2 = weight2.grad

# Update the weights using the gradients and a learning rate.
# We can use the .data attribute of the weight and bias tensors to update their values.

lr = 0.01

weight0 = weight0 - lr * grads0
weight1 = weight1 - lr * grads1
weight2 = weight2 - lr * grads2

"""
    Der SDG (Stochastic Gradient Descent) Optimizer ist eine Methode, 
    um die Gewichte und Biases eines Modells zu aktualisieren. 
    Der SDG Optimizer verwendet die Gradienten der Gewichte und Biases, 
    um die Werte der Gewichte und Biases zu aktualisieren. 
    Der SDG Optimizer verwendet eine Lernrate, um zu bestimmen, 
    wie stark die Gewichte und Biases aktualisiert werden sollen.
"""

criterion = CrossEntropyLoss()

target = F.one_hot(torch.tensor([1]), num_classes=2).float()

# torch.randn(1, 16) generates a random tensor of shape (1, 16) with values drawn from a normal distribution. This tensor is used as input to the model to generate predictions.
pred = model(torch.randn(1, 16))

# Create an optimizer object
# The optimizer is responsible for updating the weights and biases of the model during training. In this case, we are using the Stochastic Gradient Descent (SGD) optimizer, 
# which updates the weights and biases based on the gradients computed during backpropagation.
optimizer = optim.SGD(model.parameters(), lr=0.01)

loss = criterion(pred, target)
# loss.backward() computes the gradients of the loss with respect to the model parameters (weights and biases) using backpropagation. 
# This is a crucial step in training neural networks, as it allows us to update the model parameters in the direction that minimizes the loss.
loss.backward()

optimizer.step() # optimizer.step() updates the model parameters (weights and biases) using the gradients computed during backpropagation. This is where the actual learning happens, as the model parameters are adjusted to reduce the loss.