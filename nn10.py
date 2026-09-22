import torch
import torch.nn as nn

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# animal dataset
animals = pd.DataFrame({
    'name': ['cat', 'dog', 'fish', 'bird'],
    'legs': [4, 4, 0, 2],
    'weight': [4.5, 20.0, 0.1, 0.5],
    'height': [25.0, 60.0, 5.0, 15.0],
    'type': ['mammal', 'mammal', 'fish', 'bird']
})

X = animals.iloc[:, 1:-1].to_numpy()
y = animals.iloc[:, -1].to_numpy()

# Create a TensorDataset from the input features and labels
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

# Print the first sample from the dataset
input_sample, label_sample = dataset[0]

# batch size of 2, shuffle the dataset
# batch size of 2 means that the DataLoader will return batches of 2 samples at a time.
# batch is the number of samples that will be passed through the model at once. 
# A smaller batch size means that the modell will be updated more frequently,
# but it will also take longer to train the model. A larger batch size means that the model will be updated less frequently, but it will also take less time to train the model.
# sample is a single data point from the dataset.
# The Dataloader will return batches of samples from the dataset,
# where each batch will contain a specified number of samples (batch size) and the samples will be shuffled randomly.
# shuffle=True means that the samples will be shuffled randomly before being returned in batches.
# It is important to shuffle the samples before training the model, because it helps to prevent the model from learning any patterns in the order of the samples, which could lead to overfitting.
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Iterate through the dataloader
for batch_inputs, batch_labels in dataloader:
    # Print the batch inputs and labels

    print("Batch Inputs: ", batch_inputs)
    print("Batch Lables: ", batch_labels)