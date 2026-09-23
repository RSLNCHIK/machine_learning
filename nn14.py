"""
    Die Leistung des Modells bewerten.
    Auswertungsschleife schreiben.
    Wir werden die Leistung des Modells bewerten, indem wir die
    Vorhersagen des Modells mit den tatsaechlichen Werten vergleichen und
    die Genauigkeit des Modells berechnen.
"""

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import torchmetrics

model = nn.Sequential(
    nn.Linear(8, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1)
    )

# Set the modeel to evaluation mode
# Because we are evaluating the model, we do not want to update the weights and biases of the model.
model.eval()  # Set the model to evaluation mode

validation_loss = 0.0

validation_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                                  [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                                                  [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                                                  [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]], dtype=torch.float32),
                                           torch.tensor([[1.5], [2.5], [3.5], [4.5]], dtype=torch.float32)),
    batch_size=2,
    shuffle=True
)

dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                                  [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                                                  [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                                                  [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]], dtype=torch.float32),
                                           torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)),
    batch_size=2,
    shuffle=True
)

criterion = CrossEntropyLoss()

# torch.no_grad() is a context manager that disables gradient calculation
with torch.no_grad():  # Disable gradient calculation
    # labels are the actual values of the data, and features are the input
    # data that we will use to make predictions.
    for features, labels in validation_dataloader:
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        # Add the loss to the validation loss
        validation_loss += loss.item()

# Calculate the average validation loss
validation_loss_epoch = validation_loss / len(validation_dataloader)

# Set the model back to training mode
model.train()


# Create a metric to evalute the performance of the model.
# We will use the accuracy metric to evaluate the performance of the model.

metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)

for features, labels in dataloader:
    outputs = model(features)

    # Calculate the accuracy of the model over the batch of data.
    # labels.argmax(dim=-1) is used to convert the one-hot encoded labels to the
    # actual class labels. The argmax function returns the index of the maximum
    # value in the tensor, which corresponds to the class label. dim=-1 specifies
    # that we want to perform the argmax operation along the last dimension of the
    # tensor, which is the dimension that contains the class labels.

    metric.update(outputs, labels.argmax(dim=-1))

# Calculate the accuracy of the model over the entire dataset.

accuracy = metric.compute()

# Reset the metric for the next evaluation.
metric.reset()
# We use it to reset the state of the metric so hard that it can be used to 
# evaluate the performance of the model on a new dataset without any interface
# issues.

"""
    Ueberanpassung (Overfitting) ist ein Problem, das auftritt, wenn ein Modell zu gut
    auf die Trainingsdaten passt und dadurch die Leistung auf den Validierungs-
    oder Testdaten verschlechtert. Dies kann passieren, wenn das Modell zu viele
    Parameter hat oder wenn das Modell zu lange trainiert wird.
    Mit Dropout kann man Ueberanpassung verhindern, indem man zufaellig einige 
    Neuronen im Modell deativiert. Dies zwingt das Modell,
    robustere Merkmale zu lernen, die nicht von einzelnen Neuronen abhaengen.
"""
model = nn.Sequential(
    nn.Linear(8, 6),
    nn.Linear(6, 4),
    nn.Dropout(p=0.5))

outfput_train = model(dataloader)

# Set the model to evaluation mode
model.eval()

output_eval = model(dataloader)

"""
    Überanpassung liegt vor, wenn das Modell im Validierungsdatensatz 
    schlechter abgeschnitten wird als im Trainingsdatensatz.
    Die Datenerweiterung kann die Überanpassung reduzieren, 
    indem der Trainingsdatensatz künstlich erhöht wird.
    Eine Dropout-Schicht mit einer Wahrscheinlichkeit von mehr als 0 hilft, 
    die Überanpassung zu reduzieren.
"""

