import gym
import tensorflow as tf


class A3CModel(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        # Determine whether input is continuous or discrete. Generally, for
        # discrete actions, we will take the softmax of the output
        # probabilities and for the continuous we will use the linear output,
        # rescaled to the action space.
        # 
        # This will be handled by the actors themselves though, not the model,
        # so here we just determine how many output nodes to use.
        self.action_is_continuous = False
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_size = env.action_space.n
        else:
            self.action_is_continuous = True
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            action_size = env.action_space.low.shape[0]

        # Policy layer
        self.policy_dense1 = tf.keras.layers.Dense(16, activation="relu")

        if self.action_is_continuous:
            # According to A3C paper, they use a linear and softplus layer for
            # continuous output to emulate mu and sigma^2 (mean and variance)
            # of a normal distribution. To make things simple, I'll emulate
            # standard deviation sigma instead of sigma^2. This should be
            # interesting...
            self.policy_output1 = tf.keras.layers.Dense(action_size, activation='tanh')
            self.policy_output2 = tf.keras.layers.Dense(action_size, activation='softplus')
        else:
            self.policy_output1 = tf.keras.layers.Dense(action_size)

        # Value layers
        self.value_dense1 = tf.keras.layers.Dense(16, activation="relu")
        self.value_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # Forward pass on the two paths on the network
        x = self.policy_dense1(inputs)
        logits1 = self.policy_output1(x)

        v1 = self.value_dense1(inputs)
        values = self.value_output(v1)

        if self.action_is_continuous:
            logits2 = self.policy_output2(x)
            return (logits1, logits2), values

        return logits1, values