import tensorflow as tf
from utils_simulation import set_reproducibility
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import GlorotUniform, Zeros
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Rescaling,
)
import numpy as np
import random


class FedAvg(tf.keras.Model):

    def __init__(self, num_classes=4, random_seed=42):
        super(FedAvg, self).__init__()
        set_reproducibility(random_seed)


        # Layers
        self.glorot_initializer = GlorotUniform(seed=random_seed)
        self.rescale_layer = Rescaling(1.0 / 255)
        self.conv1 = Conv2D(
            16,
            3,
            padding="same",
            activation="relu",
            kernel_initializer=self.glorot_initializer)
        self.pool = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(32, 3, padding="same", activation="relu", kernel_initializer=self.glorot_initializer)
        self.conv3 = Conv2D(
            64,
            3,
            padding="same",
            activation="relu",
            kernel_initializer=self.glorot_initializer,
        )
        self.flatten_layer = Flatten()
        self.dense1 = Dense(
            128,
            activation="relu",
            kernel_initializer=self.glorot_initializer,
            kernel_constraint=MaxNorm(3),
        )
        self.dropout_layer = Dropout(0.3, seed=random_seed)
        self.dense2 = Dense(
            num_classes,
            activation="softmax",
            kernel_initializer=self.glorot_initializer,
        )

    def call(self, inputs):

        x = self.rescale_layer(inputs)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.flatten_layer(x)
        x = self.dense1(x)
        x = self.dropout_layer(x)
        x = self.dense2(x)

        return x

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = tf.keras.losses.categorical_crossentropy(labels, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def val_step(self, x_val, y_val):
        predictions = self(x_val, training=False)  # Predict using the model

        # Calculate loss (you can use a different metric if needed)
        loss = tf.keras.losses.categorical_crossentropy(y_val, predictions)

        return loss

