import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import Model

from garnn.layers.gat import GraphAttentionHead


class AttentionMechanism(layers.Layer):
    """[summary]

    [description]

    Extends:
        Model
    """

    def __init__(
        self,
        F: int,
        adjacency_matrix: np.ndarray,
        num_heads: int = 1,
        use_reverse_diffusion: bool = True,
        use_bias: bool = True,
    ):
        super(AttentionMechanism, self).__init__(name="AttentionMechanism")

        # Number of hidden units for Attention Mechanism
        self.F = F

        # store adjacency_matrix
        self.E = adjacency_matrix

        # number of attention heads
        self.num_heads = num_heads

        # populated by GraphAttentionHead layers
        self.attn_heads = []

        for _ in range(num_heads):

            # get Graph Attention Head layer
            attn_head = GraphAttentionHead(
                F=F,
                adjacency_matrix=self.E,
                use_bias=use_bias,
                use_reverse_diffusion=use_reverse_diffusion,
            )

            self.attn_heads.append(attn_head)

    def call(self, inputs):

        attention_layers = []
        # connect all attention layers to input
        for layer in self.attn_heads:

            attention_layers.append(layer(inputs))

        # now average all their outputs
        return layers.Average()(attention_layers)
