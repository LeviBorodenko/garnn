import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Layer

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


class GraphAttentionHead(Layer):
    """Returns an attention matrix based on the graph signal.
    Corresponds to one attention head.

    [description]
    Based on https://arxiv.org/pdf/1710.10903.pdf


    Arguments:
        F {int} -- dimension of internal embedding
        adjacency_matrix {np.ndarray} -- adjacency matrix of graph

    Keyword Arguments:
        use_bias {bool} -- whether to use bias or not (default: {True})
        use_reverse_diffusion {bool} -- if True, we will return 2 attention heads,
        one for the inbound process (A_in in the paper) and one for the out-bound process,
        (A_out in the paper).
        kernel_initializer {str} -- (default: {"glorot_uniform"})
        bias_initializer {str} --  (default: {"zeros"})
        attn_vector_initializer {str} -- (default: {"glorot_uniform"})
        kernel_regularizer {[type]} -- (default: {None})
        bias_regularizer {[type]} -- (default: {None})
        attn_vector_regularizer {[type]} -- (default: {None})

    Extends:
        Layer
    """

    def __init__(
        self,
        F: int,
        adjacency_matrix: np.ndarray,
        use_bias: bool = True,
        use_reverse_diffusion: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_vector_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_vector_regularizer=None,
        **kwargs
    ):
        super(GraphAttentionHead, self).__init__(**kwargs)

        # Number of features we extract and then
        # recombine to generate the attention.
        # (F` in paper)
        self.F = F

        # storing adjacency_matrix as E
        self.E = tf.constant(adjacency_matrix, dtype="float32")

        # get number of nodes from adjacency_matrix
        self.N = self.E.shape[-1]

        # whether or not to add bias after generating the F features
        self.use_bias = use_bias

        # whether to return the reversed process attention or
        # just 2 copies of the same
        self.use_reverse_diffusion = use_reverse_diffusion

        # if attention matrix is symmetric, then the reversed process
        # is the same as the forward process

        # count how many elements differ between E and E^T
        comparision = tf.equal(self.E, tf.transpose(self.E))
        num_diff_entries = int(tf.math.count_nonzero(comparision))

        # if E = E^T
        if num_diff_entries == self.N ** 2:

            # no point using reverse process
            self.use_reverse_diffusion = False

        # storing initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_vector_initializer = initializers.get(attn_vector_initializer)

        # storing regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_vector_regularizer = regularizers.get(attn_vector_regularizer)

    def build(self, input_shape):
        # we expect the input to be a
        # Extracting dimensions of graph signal
        # Number of features per node
        self.K = input_shape[-1]

        # Check # nodes agrees with adjacency_matrix
        assert self.N == input_shape[-2]

        # Check if we have a time series of graph signals
        if len(input_shape) > 3:
            self.is_timeseries = True

        # initializing kernel
        # W in paper
        self.W = self.add_weight(
            shape=(self.K, self.F),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="attn_kernel",
        )

        # initializing Bias
        if self.use_bias:
            self.B = self.add_weight(
                shape=(self.N, self.F),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                name="attn_bias",
            )

        # in the paper we need to calculate
        # [X_i*W || X_j*W] v were v is 2F dimensional
        # we skip the concatenation by decomposing v into
        # v1, v2 in R^F and thus writing the above as
        # X_i*W*v1 + X_j*W*v2
        self.v_1 = self.add_weight(
            shape=(self.F, 1),
            initializer=self.attn_vector_initializer,
            regularizer=self.attn_vector_regularizer,
            name="attn_vector_1",
        )
        self.v_2 = self.add_weight(
            shape=(self.F, 1),
            initializer=self.attn_vector_initializer,
            regularizer=self.attn_vector_regularizer,
            name="attn_vector_2",
        )

    def get_attention(self, graph_signal, adjacency_matrix):

        # If X is the graph signal then note that
        # doing the following is equivalent to the matrix in eq (1)
        # :

        # 1. calculate X*W where X is the (N x K) graph signal
        # and W is the (K x F) kernel matrix
        # 2. let v1 and v2 be two (F x 1) vectors and find
        # d1 = X*W*v1, d2 = X*W*v2 (N x 1)
        # 3. Using numpys broadcasting rules we now calculate
        # A = d1 + d2^T which will be (N x N)

        # graph signal with K features on N nodes
        X = graph_signal

        # 1.
        # Affine project each feature from R^K to R^F
        # using the kernel (W in paper) and some bias if needed.
        if self.use_bias:
            proj_X = tf.matmul(X, self.W) + self.B
        else:
            proj_X = tf.matmul(X, self.W)

        # 2.
        # multiply with v1 and v2
        d1 = tf.matmul(proj_X, self.v_1)  # (N x 1)

        d2 = tf.matmul(proj_X, self.v_2)

        # 3.
        # create an (N x N) matrix of pairwise sums of entries from
        # d1 and d2.
        # We utilise numpy broadcasting to achieve that
        # Note: we need to specify that we only transpose
        # the last 2 dimensions of d2 which due to batch-wise
        # data can have 3 dimensions.
        if self.is_timeseries:
            A = d1 + tf.transpose(d2, perm=[0, 1, 3, 2])
        else:
            A = d1 + tf.transpose(d2, perm=[0, 2, 1])

        # The above A is the unnormalized attention matrix.
        # first we remove all entries in A that correspond to edges that
        # are not in the graph.
        A = tf.multiply(A, adjacency_matrix)

        # apply non linearity (as in paper: LeakyReLU with a=0.2)
        A = tf.nn.leaky_relu(A, alpha=0.2)

        # now we softmax this matrix over its columns to normalise it.
        A = tf.nn.softmax(A)

        return A

    def call(self, inputs):

        # get graph signal from inputs
        X = inputs

        # check if we aren't using the reverse diffusion process
        if not self.use_reverse_diffusion:

            #  only calculate A_in
            A_in = self.get_attention(X, self.E)

            return A_in

        else:

            # calculate A_in and A_out
            A_in = self.get_attention(X, self.E)

            # A_out is just as A_in but we use the transpose of the
            # adjacency matrix
            E_t = tf.transpose(self.E)

            A_out = self.get_attention(X, E_t)

            # stack on the first non batch dimension
            # to obtain (batch, 2, N, N) shape
            return tf.stack([A_in, A_out], -3)

    def compute_output_shape(self, input_shape):

        # if true, we return A_in and A_out
        if self.use_reverse_diffusion:
            return 2, self.N, self.N

        # only return A_in
        else:
            return self.N, self.N
