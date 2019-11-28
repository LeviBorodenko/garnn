import datetime
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from garnn.components.attention import AttentionMechanism
from garnn.layers.gat import GraphAttentionHead, test_dim

# mute printing to termianl
# sys.stdout = open(os.devnull, "w")


# creating train data
x_train = np.random.normal(size=(1, 10, 2))

y_train = np.ones((1, 2, 10, 10))

E = np.random.randint(0, 2, size=(10, 10))


# creating data
x_val = np.random.normal(size=(100, 2), scale=100)

y_val = (x_val[:, 1] ** 2 + x_val[:, 0] ** 2) ** 0.5


inp = layers.Input(shape=(10, 2))

output = AttentionMechanism(5, E, 5)(inp)

model = keras.Model(inputs=inp, outputs=output)
# Specify the training configuration (optimizer, loss, metrics)
model.compile(
    optimizer="SGD",  # Optimizer
    # Loss function to minimize
    loss="MSE",
    metrics=["mae"],
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, update_freq="batch"
)


model.fit(
    x_train, y_train, batch_size=50, epochs=10, callbacks=[tensorboard_callback],
)
# validation_data=(x_val, y_val))
