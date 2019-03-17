"""Shared actor code that can be used both by the master and worker agents."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def rescale_mean(model, mean, old_min=-1., old_max=1.):
    return (model.action_high - model.action_low) * (mean - old_min) / (old_max - old_min) + model.action_low


def continuous_action(model, logits):
    # We would probably optimize this slightly by using the standard
    # deviation, sigma, directly, instead of having to take the square root
    # here. But that's for another day.
    mean, stddev = logits
    mean = rescale_mean(model, mean)
    normal_dist = tfp.distributions.Normal(mean[0], stddev[0])
    return tf.clip_by_value(normal_dist.sample(1), model.action_low, model.action_high)

def discrete_action(model, logits):
    return tf.nn.softmax(logits)

def act(model, state, deterministic=False):
    # When performing an action, we're not interested in the value part of the output
    # It is assumed that the model outputs a discrete action space for now.
    logits, _ = model(tf.convert_to_tensor([state], dtype=tf.float32))

    if model.action_is_continuous:
        # Note: Assumes a single action currently.
        action = continuous_action(model, logits)
        return action.numpy()[0]

    action_probabilities = discrete_action(model, logits).numpy()[0]

    if deterministic:
        return np.argmax(action_probabilities)

    # When we act according to the softmax probabilities, we are always
    # exploring to some extent, unless the agent is absolutely certain
    # that some action is the best.
    # We are also assuming that action are 0, 1, 2 and so forth of
    # course...
    return np.random.choice(len(action_probabilities), p=action_probabilities)
