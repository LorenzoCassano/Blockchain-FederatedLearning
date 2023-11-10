import tensorflow as tf

class MyModel(tf.keras.model):
        def __init__(self):
            super(MyModel, self).__init__()

            # Define the layers and architecture of your model here
            self.input = tf.keras.layers.InputLayer()
            self.rescaling = tf.keras.layers.Rescaling()
            self.conv1 = tf.keras.layers.Conv2D(16,3, padding="same", activation="relu",kernel_initializer=GlorotUniform(seed=RANDOM_SEED))

        def call(self, inputs):
            # Define the forward pass of your model
            x = self.flatten(inputs)
            x = self.dense1(x)
            x = self.dense2(x)
            return x



