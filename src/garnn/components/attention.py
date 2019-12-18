import numpy as np
import tensorflow.keras.layers as layers

from garnn.layers.gat import GraphAttentionHead

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


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

    Layer-inputs:
         X (batch, timesteps, N, F), graph signals on N nodes with F
            features. Also works with individual graph signals with no
            time steps, i.e. (batch, N, F).

    Layer-outputs:
        A (batch, timesteps, 2, N, N), 2 attention matrices (if we use
        the reverse process) for each graph signal. If we do not use
        the reverse process, e.q. when the adjacency matric is symmetric
        then we get (batch, timesteps, N, N).
        If we do not have a time series then (batch, 2, N, N) or (batch, N, N)
        is returned.

    Example:

        # creating train data
        x_train = np.random.normal(size=(1000, 10, 10, 2))
        y_train = np.ones((1000, 10, 2, 10, 10))

        # random adjacency_matrix
        E = np.random.randint(0, 2, size=(10, 10))

        # building tiny model
        X = layers.Input(shape=(None, 10, 2))
        A = AttentionMechanism(F=5,
                               adjacency_matrix=E,
                               num_heads=5,
                               use_reverse_diffusion=True)(X)

        model = keras.Model(inputs=X, outputs=A)

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
