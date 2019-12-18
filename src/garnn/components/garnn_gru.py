import tensorflow as tf
import tensorflow.keras.layers as layers

from garnn.layers.diffconv import GraphDiffusionConvolution

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


class GRUCell(layers.Layer):
    """Cell class for the GARNN-GRU recurrent layer.
    based on https://milets19.github.io/papers/milets19_paper_8.pdf
    and in part https://arxiv.org/pdf/1707.01926.pdf.

    [description]

    Arguments:
        num_nodes {int} -- Need to provide the number of nodes due to
        keras demanding to know the shape of the hidden state of the RNN
        layer before self.build() is run. Use the GARNN_GRU wrapper to
        avoid this inconvienience.
        num_hidden_features {int} -- Number of hidden features (Q in paper)
        **kwargs

    Keyword Arguments:
        num_diffusion_steps {int} -- Number of diffusion steps used
        in ALL diffusion. (default: {5})

    """

    def __init__(
        self,
        num_nodes: int,
        num_hidden_features: int,
        num_diffusion_steps: int = 5,
        **kwargs
    ):
        super(GRUCell, self).__init__(**kwargs)

        # number of hidden features for the GRU mechanism
        assert num_hidden_features > 0
        self.num_hidden_features = num_hidden_features

        # short-cut to the number of hidden features
        self.F_h = num_hidden_features

        # getting number of nodes N for the required
        # state_size attribute. We need to do it in __init__
        # as it needs to be defined before build is called.
        self.N = num_nodes

        # Number of diffusion steps. For now we'll use the same
        # number for all 3 convolutions. (K in paper)
        assert num_diffusion_steps > 0
        self.K = num_diffusion_steps

        self.state_size = tf.TensorShape([self.N, self.F_h])

    def build(self, input_shape):

        # We expect to receive (X, A)
        # A - Attention (may be 2 matrices if we use reverse
        #                diffusion) (N, N) or (2, N, N)
        # X - graph signal (N, F)

        # getting number of nodes N, again.
        x_shape = input_shape[0]
        self.N = x_shape[-2]

        self.state_size = tf.TensorShape([self.N, self.F_h])

        # Initiating biases
        self.bias_r = self.add_weight(
            shape=(self.N, self.F_h), initializer="zeros", name="reset_gate_bias"
        )

        self.bias_u = self.add_weight(
            shape=(self.N, self.F_h), initializer="zeros", name="update_gate_bias"
        )

        self.bias_c = self.add_weight(
            shape=(self.N, self.F_h), initializer="zeros", name="cell_bias"
        )

        # Initiating Convolutions
        self.conv_r = GraphDiffusionConvolution(
            features=self.F_h, num_diffusion_steps=self.K, name="Reset_gate_convolution"
        )

        self.conv_u = GraphDiffusionConvolution(
            features=self.F_h,
            num_diffusion_steps=self.K,
            name="Update_gate_convolution",
        )

        self.conv_c = GraphDiffusionConvolution(
            features=self.F_h, num_diffusion_steps=self.K, name="Cell_convolution"
        )

    def call(self, inputs, states):

        # get graph signal X and attention tensor A
        X, A = inputs

        # get previous hidden state
        # we reshape because for some reason H has shape (1, batch, N, F_h)
        # instead of the required (batch, N, F_h)
        H = tf.reshape(states, (-1, self.N, self.F_h))

        # concatinate graph signal and hidden state
        # We need this for some calculations. See 2.3
        # in the paper.
        X_H = tf.concat([X, H], axis=-1)

        # calulating reset gate r^(t)
        gate_r = tf.sigmoid(self.conv_r((X_H, A)) + self.bias_r)

        # update gate
        gate_u = tf.sigmoid(self.conv_u((X_H, A)) + self.bias_u)

        # calculating cell state C^(t)
        # apply reset gate
        H_resetted = tf.multiply(gate_r, H)

        # fuse with graph signal
        X_H_resetted = tf.concat([X, H_resetted], axis=-1)

        # final step
        cell_state = tf.tanh(self.conv_c((X_H_resetted, A)) + self.bias_c)

        # Updating hidden state
        H = tf.multiply(gate_u, H) + tf.multiply(1 - gate_u, cell_state)

        return H, H


class garnn_gru(layers.Layer):
    """The modified GRU used in GARNNS.
    Based on https://milets19.github.io/papers/milets19_paper_8.pdf
    and in part https://arxiv.org/pdf/1707.01926.pdf.

    [description]

    Layer-inputs:
        Tuple contraining:
            1. X (batch, timesteps, N, F), graph signals on N nodes with F
            features.
            2. A, output of an AttentionMechanism layer based on X.

            IN THE FOLLOWING ORDER: (X, A)

    Layer-outputs:
        (Final) Hidden State of the GRU. H in paper. Has shape
        (batch, N, num_hidden_features)

    Example:

        # creating train data
        x_train = np.random.normal(size=(1000, 10, 10, 2))
        y_train = np.ones((1000, 10, 1))

        # random adjacency_matrix
        E = np.random.randint(0, 2, size=(10, 10))

        # building tiny model
        X = layers.Input(shape=(None, 10, 2))
        A = AttentionMechanism(F=5,
                               adjacency_matrix=E,
                               num_heads=5,
                               use_reverse_diffusion=True)(X)

        output = garnn_gru(num_hidden_features=1)((X, A))

        model = keras.Model(inputs=X, outputs=output)

    Arguments:
        num_hidden_features {int} -- Number of hidden features (Q in paper)
        **kwargs

    Keyword Arguments:
        num_diffusion_steps {int} -- Number of diffusion steps used
        in ALL diffusion. (default: {5})

    """

    def __init__(
        self, num_hidden_features: int, num_diffusion_steps: int = 5, **kwargs
    ):
        super(garnn_gru, self).__init__()

        # storing parameters that we'll pass to the cell
        self.F_h = num_hidden_features
        self.K = num_diffusion_steps

        self.kwargs = kwargs

    def build(self, input_shape):
        # We expect to receive (X, A)
        # A - Attention (may be 2 matrices if we use reverse
        #                diffusion) (, N, N) or (, 2, N, N)
        # X - graph signal (, N, F)

        # getting number of nodes N, again.
        x_shape = input_shape[0]
        self.N = x_shape[-2]

        # d
        self.cell = GRUCell(self.N, self.F_h, self.K)

        self.RNN = layers.RNN(self.cell, self.kwargs)

    def call(self, inputs):
        # This class is simply a wrapper that creates the
        # RNN based on the GRUCell and relays the inputs into
        # it.
        return self.RNN(inputs)
