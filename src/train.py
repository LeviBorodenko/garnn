import datetime
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from garnn.components.attention import AttentionMechanism
from garnn.layers.diffconv import DiffuseFeatures

# mute printing to termianl
# sys.stdout = open(os.devnull, "w")


# creating train data
x_train = np.random.normal(size=(1000, 10, 2))
y_train = np.ones((1000, 2))
E = np.random.randint(0, 2, size=(10, 10))

# building test model
X = layers.Input(shape=(10, 2))
A = AttentionMechanism(5, E, 5, use_reverse_diffusion=False)(X)
output = DiffuseFeatures(3)((X, A))

model = keras.Model(inputs=X, outputs=output)

# Specify the training configuration (optimizer, loss, metrics)
model.compile(
    optimizer="SGD",
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