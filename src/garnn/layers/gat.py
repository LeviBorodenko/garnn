import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Layer


class GraphAttentionHead(Layer):
    """Returns a unnormalized attention matrix based on the graph signal.
    Corresponds to one attention head.

    [description]
    Based on https://arxiv.org/pdf/1710.10903.pdf

    Extends:
        Layer
    """

    def __init__(
        self,
        F: int,
        use_bias: bool = True,
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

        # whether or not to add bias after generating the F features
        self.use_bias = use_bias

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

        # Number of nodes
        self.N = input_shape[-2]

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

    def call(self, inputs):

        # If X is the graph signal then note that
        # doing the following is equivalent to the matrix in eq (1)
        # :

        # 1. calculate X*W where X is the (N x K) graph signal
        # and W the (K x F) kernel matrix
        # 2. let v1 and v2 be two (F x 1) vectors and find
        # d1 = X*W*v1, d2 = X*W*v2 (N x 1)
        # 3. Using numpys broadcasting rules we now calculate
        # A = d1 + d2^T which will be (N x N)

        # graph signal with K features on N nodes
        X = inputs

        # 1.
        # Affine project each feature from R^K to R^F
        # using the kernel (W in paper) and some bias if needed.
        if self.use_bias:
            proj_X = tf.matmul(X, self.W) + self.B
        else:
            proj_X = tf.matmul(X, self.W)

        # 2.
        # multiply with v1 and v2
        d1 = tf.matmul(proj_X, self.v1)  # (N x 1)

        d2 = tf.matmul(proj_X, self.v2)

        # 3.
        # create an (N x N) matrix of pairwise sums of entries from
        # d1 and d2.
        # We utilise numpy broadcasting to achieve that
        A = d1 + tf.transpose(d2)

        # The above A is the unnormalized attention matrix. Which
        # is exactly what this layer is supposed to output
        return A

    def compute_output_shape(self, input_shape):

        # return (N x N) attention matrix
        return self.N, self.N


class AdjacencyFeeder(Layer):
    """Once initiated with the adjacency matrix of your network,
    this layer is used to access that matrix later on.

    Arguments:
        adjacency_matrix {np.ndarray} -- adjacency matrix

    Extends:
        Layer
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        super(AdjacencyFeeder, self).__init__()

        # store adjacency matrix on initiation
        self.E = tf.constant(adjacency_matrix)

    def call(self, inputs):

        # return stored matrix on call
        return self.E


class AttentionNormaliser(Layer):
    """Takes an unnormalised attention matrix and normalises it
    using the adjacency matrix.

    Let Â be the unnormalised attention,
    E is the adjacency matrix,
    * is the Hadamard product operation and
    softmax() is the softmax operation applied to each column of a
    matrix.
    Then the normalised attention is given by:

    A = softmax(Â * E)

    Note that this is a transition matrix and can thus be used in a
    diffusion process.

    Extends:
        Layer
    """

    def __init__(self, arg):
        super(AttentionNormaliser, self).__init__()
        self.arg = arg
