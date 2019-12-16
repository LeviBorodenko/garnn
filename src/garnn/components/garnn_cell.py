import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import activations

from garnn.components.attention import AttentionMechanism
from garnn.layers.diffconv import GraphDiffusionConvolution


class GRUCell(layers.Layer):
    """Cell class for the GRU layer.
    based on https://arxiv.org/pdf/1707.01926.pdf

    """

    def __init__(
        self,
        num_hidden_features: int,
        num_diffusion_steps: int = 5,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias: bool = True,
        **kwargs
    ):
        super(GRUCell, self).__init__(**kwargs)

        # number of hidden features for the GRU mechanism
        assert num_hidden_features > 0
        self.num_hidden_features = num_hidden_features

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        # short-cut to the number of hidden features
        self.F_h = num_hidden_features

        # Number of diffusion steps. For now we'll use the same
        # number for all 3 convolutions. (K in paper)
        assert num_diffusion_steps > 0
        self.K = num_diffusion_steps

    def build(self, input_shape):

        # We expect to receive (X, A)
        # A - Attention (may be 2 matrices if we use reverse
        #                diffusion) (N, N) or (2, N, N)
        # X - graph signal (N, F)

        # getting number of nodes N
        x_shape = input_shape[0]
        self.N = x_shape[-2]

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
        H = states[0]

        # concatinate graph signal and hidden state
        tf.concat()
