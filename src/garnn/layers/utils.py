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
