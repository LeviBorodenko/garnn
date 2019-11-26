import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from layers.utils import Linear

# create simple 1 layer perceptron
in_p = layers.Input(shape=(2))
x = Linear(5)(in_p)
x = layers.Activation("relu")(x)
out_p = Linear(1)(x)

# create model
mlp = keras.Model(inputs=in_p, outputs=out_p)
