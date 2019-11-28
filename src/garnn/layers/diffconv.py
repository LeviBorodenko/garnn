"""Graph Diffusion RNN

Based on https://arxiv.org/pdf/1707.01926.pdf
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import constraints, initializers, regularizers


class DiffuseFeatures(layers.Layer):
    """Applies diffusion graph convolution given a
    graph signal X and an attention matrix A.

    Procedure is based on https://arxiv.org/pdf/1707.01926.pdf

    Arguments:
        num_diffusion_steps {int} -- K in paper, number of steps
        in the diffusion process.

    Keyword Arguments:
        theta_initializer {str} -- (default: {"ones"})
        theta_regularizer {[type]} -- (default: {None})
        theta_constraint {[type]} -- (default: {None})
    """

    def __init__(
        self,
        num_diffusion_steps: int,
        theta_initializer="ones",
        theta_regularizer=None,
        theta_constraint=None,
    ):
        super(DiffuseFeatures, self).__init__()

        # number of diffusino steps (K in paper)
        self.K = num_diffusion_steps

        assert self.K > 0

        # get regularizer, initializer and constraint for theta
        self.theta_initializer = initializers.get(theta_initializer)
        self.theta_regularizer = regularizers.get(theta_regularizer)
        self.theta_constraint = constraints.get(theta_constraint)

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
        # tf.polyval needs a list of tensors as the coeff. so ww
        # unstack theta
        diffusion_matrix = tf.math.polyval(tf.unstack(self.theta), A)

        # apply it to X to get a matrix C = [C_1, ..., C_F] (N x F)
        # of diffused features
        diffused_features = tf.matmul(diffusion_matrix, X)

        # now we add all diffused features (columns of the above matrix)
        # and apply a non linearity to obtain H:,q (eq. 3 in paper)
        H = tf.math.reduce_sum(diffused_features, axis=1)
        H = tf.sigmoid(H)

        return H
