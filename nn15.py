"""
    Die Leistung des Modells verbessern.
    Schritt1: Ueberpasse den Traininssatz
    Schritt2: Overfitting verhindern
    Schritt3: Optimierung der Hyperparameter
    Regulisierungstechniken wie Dropout, L1/L2-Regularisierung und Datenaugmentation können helfen, 
    Overfitting zu verhindern.
"""
import torch
import torch.nn as nn
import numpy as np

for idx in range(1, 10):
    # Randomly select a learning rate between 1e-5 and 1e-1
    factor = np.random.uniform(2, 4)
    lr = 10 * (-factor)

    # Randomly select a momentum between 0.85 and 0.99
    momentum = np.random.uniform(0.85, 0.99)

