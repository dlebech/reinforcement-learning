import numpy as np
import tensorflow as tf

from rl.agent.a3c import model


def test_model_variable_scope(cartpole_env):
    a3c_model = model.A3CModel(cartpole_env)
    state = cartpole_env.reset()
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
    a3c_model(state_tensor)

    value_weights = [v for v in a3c_model.trainable_weights if "value_scope" in v.name]
    actor_weights = [v for v in a3c_model.trainable_weights if "actor_scope" in v.name]

    # Two sets of weight layers, each with a bias.
    assert len(value_weights) == 4
    assert len(actor_weights) == 4


def test_model_call_discrete(cartpole_env):
    a3c_model = model.A3CModel(cartpole_env)

    # Cartpole has two actions
    state = cartpole_env.reset()
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

    # Assert it's callable
    logits, values = a3c_model(state_tensor)

    assert logits.shape == (1, 2)
    assert values.shape == (1, 1)


def test_model_call_continuous(pendulum_env):
    a3c_model = model.A3CModel(pendulum_env)

    # Pendulum has
    state = pendulum_env.reset()
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

    # Assert it's callable and the output is a tuple.
    (logits1, logits2), values = a3c_model(state_tensor)

    assert logits1.shape == (1, 1)
    assert logits2.shape == (1, 1)
    assert values.shape == (1, 1)
