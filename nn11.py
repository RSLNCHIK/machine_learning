"""
    Die erste Trainingsschleife fuer das neutrale Netzwerk.
    Wir werden die Gewichte und Biases des Modells aktualisieren,
    indem wir die Gradienten der Gewichte und Biases berechnen und die Werte der Gewichte und Biases aktualisieren.

    1. Model erstellen
    2. Loss function definieren (Verlustfunktion)
    3. Optimizer definieren (Optimierer)
    4. Trainingsschleife durchlaufen
    4.1 Verlust berechnen (Vorwaertsdurchlauf)
    4.2 Gradienten berechen (Rueckwaertsdurchlauf)
    4.3 Gewichte und Biases aktualisieren (Optimierung von Modellparametern)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

y_pred = np.array([3, 5.0, 2.5, 7.0])
y = np.array([3.0, 4.5, 2.0, 8.0])


model = nn.Sequential(
    nn.Linear(8, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1)
    )

# Calculate MSE using Numpy
mse_numpy = np.mean((y_pred - y) ** 2)

# Criterion for MSE Loss
criterion = nn.MSELoss()

# Calculate MSE using PyTorch
mse_pytorch = criterion(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

print("MSE using Numpy: ", mse_numpy)
print("MSE using PyTorch: ", mse_pytorch)

"""Eine Trainingsschleife schreiben, die die Gewichte und Biases des Modells aktualisiert."""

num_epochs = 5

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

dataset = TensorDataset(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(num_epochs):
    for data in dataloader:
        # Set the Optimizer to zero the gradients
        optimizer.zero_grad()

        # Forward pass
        # Run a forward pass of the model to get the predictions
        # The forward pass is the process of passing the input data through the model to get the predictions. 
        # The forward pass is done by calling the model with the input data as an argument. 
        # The model will then compute the predictions based on the input data and the current weights and biases of the model.
        # inputs, targets = data is a tuple that contains the input data and the target data. 
        # The input data is the data that is passed through the model to get the predictions.
        # The target data is the data that is used to calculate the loss function.
        # The target data is the data that we want the model to predict.
        inputs, targets = data
        predictions = model(inputs)

        # Calculate the loss
        loss = criterion(predictions, targets)

        # Backward pass
        # Compute the gradients of the loss with respect to the weights and biases of the model
        # the gradients of the loss is a tensor that contains the gradients of the loss with respect to the weights and biases of the model.
        # The gradients of the loss are computed by calling the backward() method of the loss tensor
        loss.backward()

        # Update the weights and biases of the model using the optimizer
        optimizer.step()



