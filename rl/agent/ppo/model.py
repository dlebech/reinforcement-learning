import gym
import tensorflow as tf

from rl.agent import util


def build_model(env: gym.Env) -> tf.keras.Model:
    i = tf.keras.layers.Input(shape=env.observation_space.shape)

    action_is_continuous, action_size, _, _ = util.parse_env(env)

    policy = tf.keras.layers.Dense(32, activation="relu", name="policy_dense1")(i)

    if action_is_continuous:
        raise NotImplementedError()
    else:
        policy_out = tf.keras.layers.Dense(
            action_size, activation="softmax", name="policy_out"
        )(policy)

    value = tf.keras.layers.Dense(32, activation="relu", name="value_dense1")(i)
    value_out = tf.keras.layers.Dense(1, name="value_out")(value)

    return tf.keras.models.Model(inputs=[i], outputs=[policy_out, value_out])
