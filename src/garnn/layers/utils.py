import tensorflow as tf
import tensorflow.keras.layers as layers


class Linear(layers.Layer):
    """Simple dense layer

    [description]

    Extends:
        layers.Layer
    """

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="Bias",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
            name="Weight_matrix",
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


def is_using_reverse_process(input_shape):
    """Check if output of attention head is a single
    Attention matrix or 2 attention matrices - one for A_in
    one for A_out

    [description]

    Arguments:
        input_shape {[tuple]} -- input_shape
    """

    # dimension of attention layer output
    dim = len(input_shape)

    # (batch, 2, N, N) if we use A_in and A_out
    if dim == 4:
        return True

    # (batch, N, N) is we aren't
    elif dim == 3:
        return False
    else:
        raise ValueError(f"Invalid attention shape {input_shape}")
