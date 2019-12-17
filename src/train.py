import datetime
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from garnn.models.garnn import garnn_model

# mute printing to termianl
# sys.stdout = open(os.devnull, "w")


# creating train data
# x_train = np.random.normal(size=(10, 12, 207, 10))
# y_train = np.ones((10, 207, 1))


# Specify the training configuration (optimizer, loss, metrics)
garnn_model.compile(
    optimizer="SGD",
    # Loss function to minimize
    loss="MSE",
    metrics=["mae"],
)

garnn_model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, update_freq="batch"
)


# garnn_model.fit(
#     x_train, y_train, batch_size=2, epochs=1, callbacks=[tensorboard_callback],
# )
