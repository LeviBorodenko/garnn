import datetime
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from models.mlp import mlp

# mute printing to termianl
# sys.stdout = open(os.devnull, "w")


# creating train data
x_train = np.random.normal(size=(5000, 2))

y_train = (x_train[:, 1] ** 2 + x_train[:, 0] ** 2) ** 0.5

# creating data
x_val = np.random.normal(size=(100, 2), scale=100)

y_val = (x_val[:, 1] ** 2 + x_val[:, 0] ** 2) ** 0.5


class ConstantLayer(layers.Layer):
    """docstring for ConstantLayer"""

    def __init__(self, const=[1, 2, 3]):
        super(ConstantLayer, self).__init__()
        self.const = tf.constant(const)

    def call(self, inputs):
        return self.const


inp = layers.Input(shape=(1))
output = ConstantLayer()(inp)

model = keras.Model(inputs=inp, outputs=output)

print(model(1))


# Specify the training configuration (optimizer, loss, metrics)
# mlp.compile(optimizer="SGD",  # Optimizer
#             # Loss function to minimize
#            loss="MSE",
#            metrics=["mae"])

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir=log_dir, histogram_freq=1, update_freq="batch")


# mlp.fit(x_train, y_train,
#         batch_size=100,
#         epochs=50,
#         callbacks=[tensorboard_callback],
#         validation_data=(x_val, y_val))
