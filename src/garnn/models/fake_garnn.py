import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from garnn.components.attention import AttentionMechanism
from garnn.layers.diffconv import GraphDiffusionConvolution

graph_signals = layers.Input(shape=(3, 1))
Attn = AttentionMechanism(F=5, adjacency_matrix=E, num_heads=1)(graph_signals)
output = GraphDiffusionConvolution(3, 3)((graph_signals, Attn))
