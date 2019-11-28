import numpy as np
import tensorflow.keras.layers as layers

from garnn.layers.gat import GraphAttentionHead


class AttentionMechanism(layers.Layer):
    """Attention Mechanism utilised by GARNNs.

    We use multiple attention heads and average their outputs.
    Takes a graph signal and returns 1 or 2 edge attention matrices.

    Arguments:
        F {int} -- Dimenions of hidden representation used for attention.
        adjacency_matrix {np.ndarray} -- adjacency matrix of graph

    Keyword Arguments:
        num_heads {int} -- Number of attenheads to be used. (default: {1})
        use_reverse_diffusion {bool} -- Whether or not to us both A_in
        and A_out or simply only A_in (default: {True})
        use_bias {bool} -- Use bias in calculating the hidden representation
        (default: {True})

    Extends:
        Layer
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

        # apply all attention layers to inputs
        for layer in self.attn_heads:

            attention_layers.append(layer(inputs))

        # now average all their outputs
        return layers.Average()(attention_layers)
