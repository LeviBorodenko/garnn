"""Reimplementation of the GARRN used by
https://milets19.github.io/papers/milets19_paper_8.pdf

[description]
For simplicity we assume that the graph signals have
207 nodes and 10 features. We also choose some random
adjacency matrix as this should only be a demonstration
of how to build your own models with the layers provided
by this repo.
"""
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from garnn.components.attention import AttentionMechanism
from garnn.components.garnn_gru import garnn_gru

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


# Creating random adjacency matrix E
E = np.random.randint(0, 2, size=(207, 207))

# Input of the 12 graph signals
X = Input(shape=(12, 207, 10))

# creating attention mechanism with 3 heads and an
# embedding size of 16
A = AttentionMechanism(16, adjacency_matrix=E, num_heads=3)(X)

# now piping X and A into the 2 GRU layers
# First layer streches the features into 64 diffused features.
# We assume the we are using 6 hop diffusions.
gru_1 = garnn_gru(num_hidden_features=64, num_diffusion_steps=6, return_sequences=True)(
    (X, A)
)

# And then we use another gru to shrink it back to 1 feature - the feature
# that we are predicting. We use the same attention for this layer, but note
# we could also introduce a new attention mechanism for the next gru.
output = garnn_gru(num_hidden_features=1, num_diffusion_steps=6)((gru_1, A))

garnn_model = Model(inputs=X, outputs=output)
