"""Graph Diffusion

Based on https://arxiv.org/pdf/1707.01926.pdf
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import constraints, initializers, regularizers

from garnn.layers.utils import is_using_reverse_process

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


class DiffuseFeatures(layers.Layer):
    """Applies diffusion graph convolution given a
    graph signal X and an attention matrix A.
    Returns a single new feature.

    Procedure is based on https://arxiv.org/pdf/1707.01926.pdf

    Arguments:
        num_diffusion_steps {int} -- K in paper, number of steps
        in the diffusion process.

    Keyword Arguments:
        theta_initializer {str} -- (default: {"ones"})
        theta_regularizer {[type]} -- (default: {None})
        theta_constraint {[type]} -- (default: {None})
        use_activation {bool} -- whether or not to apply tanh to the
        defused features (default: {False})
    """

    def __init__(
        self,
        num_diffusion_steps: int,
        theta_initializer="ones",
        theta_regularizer=None,
        theta_constraint=None,
        use_activation: bool = False,
        **kwargs
    ):
        super(DiffuseFeatures, self).__init__(kwargs)

        # number of diffusino steps (K in paper)
        self.K = num_diffusion_steps

        assert self.K > 0

        # get regularizer, initializer and constraint for theta
        self.theta_initializer = initializers.get(theta_initializer)
        self.theta_regularizer = regularizers.get(theta_regularizer)
        self.theta_constraint = constraints.get(theta_constraint)

        self.use_activation = use_activation

    def build(self, input_shape):

        # shape of graph signal (batch, N, K)
        self.signal_shape = input_shape[0]

        # shape of attention matrix (batch, N, N)
        self.attention_shape = input_shape[1]

        # save number or nodes
        self.N = self.attention_shape[-1]

        # save number of features per node
        self.F = self.signal_shape[-1]

        # we need to calculate (sum theta_k * Attention_t^k) X_t
        # (k from 0 to K-1)
        # so now we are initializing the theta vector (R^K)
        self.theta = self.add_weight(
            shape=(self.K),
            initializer=self.theta_initializer,
            regularizer=self.theta_regularizer,
            constraint=self.theta_constraint,
            name="Diffusion_Coeff",
        )

    def call(self, inputs):

        # get edge attention (A_t) and graph signal (X_t)
        X, A = inputs

        # calculate diffusion matrix: sum theta_k * Attention_t^k
        # tf.polyval needs a list of tensors as the coeff. thus we
        # unstack theta
        diffusion_matrix = tf.math.polyval(tf.unstack(self.theta), A)

        # apply it to X to get a matrix C = [C_1, ..., C_F] (N x F)
        # of diffused features
        diffused_features = tf.matmul(diffusion_matrix, X)

        # now we add all diffused features (columns of the above matrix)
        # and apply a non linearity to obtain H:,q (eq. 3 in paper)
        H = tf.math.reduce_sum(diffused_features, axis=2)
        if self.use_activation:
            H = tf.tanh(H)

        # H has shape (batch, N) but as it is the sum of columns
        # we reshape it into (batch, N, 1)
        return tf.expand_dims(H, -1)


class GraphDiffusionConvolution(layers.Layer):
    """Applies Graph Diffusion Convolution to a Graph Signal
    X and attention matrices A.

    You need to specify how many diffusion steps (K in paper) and
    features (Q in paper) you want.

    returns a (batch, N, Q) diffused graph signal

    Arguments:
        features {int} -- Q in paper
        num_diffusion_steps {int} -- K in paper
        **kwargs {[type]} -- [description]

    Keyword Arguments:
        theta_initializer {str} -- [description] (default: {"ones"})
        theta_regularizer {[type]} -- [description] (default: {None})
        theta_constraint {[type]} -- [description] (default: {None})
        use_activation {bool} -- whether or not to apply tanh to the
        defused features (default: {False})
    """

    def __init__(
        self,
        features: int,
        num_diffusion_steps: int,
        theta_initializer="ones",
        theta_regularizer=None,
        theta_constraint=None,
        use_activation: bool = False,
        **kwargs
    ):
        super(GraphDiffusionConvolution, self).__init__(kwargs)

        # number of features to generate (Q in paper)
        assert features > 0
        self.Q = features

        # number of diffusion steps for each output feature
        self.K = num_diffusion_steps

        # storing initializer, regularizer and contraints for theta
        self.theta_initializer = theta_initializer
        self.theta_regularizer = theta_regularizer
        self.theta_constraint = theta_constraint

        self.use_activation = use_activation

    def build(self, input_shape):

        # We expect to receive (X, A)
        # A - Attention (may be 2 matrices if we use reverse
        #                diffusion) (N, N) or (2, N, N)
        # X - graph signal (N, F)
        X_shape, A_shape = input_shape

        # check if singular or dual attention
        self.is_dual = is_using_reverse_process(A_shape)

        # initialise Q diffusion convolution filters
        self.filters = []

        for _ in range(self.Q):
            layer = DiffuseFeatures(
                num_diffusion_steps=self.K,
                theta_initializer=self.theta_initializer,
                theta_regularizer=self.theta_regularizer,
                theta_constraint=self.theta_constraint,
                use_activation=self.use_activation,
            )
            self.filters.append(layer)

    def apply_filters(self, X, A):
        """Applies diffusion convolution self.Q times to get a
        (batch, N, Q) diffused graph signal

        [description]

        Arguments:
            X {(batch, N, K)} -- graph signal
            A {(batch , N, N)} -- Attention matrix
        """

        # this will be a list of Q diffused features.
        # Each diffused feature is a (batch, N, 1) tensor.
        # Later we will concat all the features to get one
        # (batch, N, Q) diffused graph signal
        diffused_features = []

        # iterating over all Q diffusion filters
        for diffusion in self.filters:
            diffused_feature = diffusion((X, A))
            diffused_features.append(diffused_feature)

        # concat them into (batch, N, Q) diffused graph signal
        H = tf.concat(diffused_features, -1)

        return H

    def call(self, inputs):

        # get graph signal X and attention tensor A
        X, A = inputs

        if self.is_dual is False:

            # simply apply the self.Q filters to get (batch, N, Q)
            # diffused graph signal
            H = self.apply_filters(X, A)
            return H

        else:
            # if we have 2 attention matrices then GARRNS simply
            # get the two diffused graph signals and add them together.
            # Equation 2 in paper
            ####

            # First we get both attention matrices
            A_in, A_out = tf.unstack(A, axis=-3)

            # get both diffused graph signals
            H_in = self.apply_filters(X, A_in)
            H_out = self.apply_filters(X, A_out)

            # get their sum
            H = layers.Add()([H_in, H_out])
            return H
