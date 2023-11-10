from fedAvg import *

class FedProx(FedAvg):

    SERVER_WEIGHTS = None
    def __init__(self, num_classes=4, mu=0.01, random_seed=42):
        super(FedProx, self).__init__(num_classes, random_seed)

        self.mu = tf.constant(mu, dtype=tf.float32)

    def difference_models_norm_2(self):
        """
        Return the norm 2 difference between the two model parameters
        """

        t1 = tf.concat([tf.reshape(value, [-1]) for value in self.trainable_weights],axis=0)
        t2 = tf.concat([tf.reshape(value, [-1]) for value in FedProx.SERVER_WEIGHTS], axis=0)
        assert len(t2) != 0
        diff = tf.subtract(t1, t2)
        squared_norm = tf.square(tf.norm(diff))
        return squared_norm

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = tf.keras.losses.categorical_crossentropy(labels, predictions) + (self.mu/2 * self.difference_models_norm_2())

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

