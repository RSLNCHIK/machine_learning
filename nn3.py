# Sigmoid und Softmax Funktionen
# Sigmoid: Wird verwendet, um Werte auf einen Bereich zwischen 0 und 1 zu transformieren. Sie ist nützlich in binären Klassifizierungsproblemen.
# Softmax: Wird verwendet, um Werte auf eine Wahrscheinlichkeitsverteilung zu transformieren. Sie ist nützlich in multi-klassigen Klassifizierungsproblemen.
# Beide fuegen nichtlineare Transformationen hinzu, die es dem Modell ermoeglichen, komlexe Muster in den Daten zu lernen.

import torch
import torch.nn as nn


input_tensor = torch.tensor([[2.4]])


# Create a sigmoid activation function
sigmoid = nn.Sigmoid()

probability = sigmoid(input_tensor)

print("Probability after Sigmoid: ", probability)

# Create a softmax activation function
# Softmax wird verwendet, um die Ausgaben eines Modells in Wahrscheinlichkeiten umzuwandeln, die sich zu 1 summieren. Dies ist besonders nützlich in Klassifizierungsproblemen mit mehreren Klassen, da es ermöglicht, die relative Wahrscheinlichkeit jeder Klasse zu interpretieren.
softmax = nn.Softmax()

probabilities = softmax(input_tensor)

print("Probabilities after Softmax: ", probabilities)