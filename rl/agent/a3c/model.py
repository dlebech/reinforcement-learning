import gym
import tensorflow as tf

from rl.agent import util


class A3CModel(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        self.action_is_continuous, self.action_size, self.action_low, self.action_high = util.parse_env(
            env
        )

        # Policy (actor) layer
        self.policy_dense1 = tf.keras.layers.Dense(32, activation="relu")

        if self.action_is_continuous:
            # According to A3C paper, they use a linear and softplus layer for
            # continuous output to emulate mu and sigma^2 (mean and variance)
            # of a normal distribution. To make things simple, I'll emulate
            # standard deviation sigma instead of sigma^2. This should be
            # interesting...
            self.policy_output1 = tf.keras.layers.Dense(
                self.action_size, activation="tanh"
            )
            self.policy_output2 = tf.keras.layers.Dense(
                self.action_size, activation="softplus"
            )
        else:
            self.policy_output1 = tf.keras.layers.Dense(self.action_size)

        # Value layers
        self.value_dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.value_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # Forward pass on the two paths on the network

        with tf.variable_scope("value_scope"):
            v1 = self.value_dense1(inputs)
            values = self.value_output(v1)

        with tf.variable_scope("actor_scope"):
            x = self.policy_dense1(inputs)
            logits1 = self.policy_output1(x)

            if self.action_is_continuous:
                logits2 = self.policy_output2(x)
                return (logits1, logits2), values

        return logits1, values
