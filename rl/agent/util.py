import collections
import logging
import threading

import gym
import numpy as np

logger = logging.getLogger(__name__)


class WorkerTracker:
    def __init__(self, global_model, model_path):
        self.lock = threading.Lock()
        self.global_model = global_model
        self.model_path = model_path
        self.global_episodes = 0
        self.global_best_reward = -np.inf
        self.rewards = []
        self.losses = []
        self.window = 50

    def episode_complete(
        self, index, steps, reward, loss, policy_loss=None, value_loss=None
    ):
        with self.lock:
            self.global_episodes += 1
            self.rewards.append(reward)
            if policy_loss is not None and value_loss is not None:
                loss = [loss, policy_loss, value_loss]
            self.losses.append(loss)

        logger.debug(
            f"Episode: {self.global_episodes}, "
            f"Worker: {index}, "
            f"Steps: {steps}, "
            f"Reward: {np.round(reward, 1)}, "
            f"Moving average: {np.round(np.mean(self.rewards[-self.window:]), 1)}, "
            f"Loss: {np.round(loss, 1)}, "
            f"Moving average: {np.round(np.mean(self.losses[-self.window:]), 1)}"
        )

        if self.global_model is not None and reward > self.global_best_reward:
            logger.info(
                f"New best reward: {reward} vs. {self.global_best_reward}, "
                f"Saving model to: {self.model_path}"
            )
            with self.lock:
                self.global_model.save_weights(self.model_path)
                self.global_best_reward = reward


class WorkerMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


def parse_env(env):
    """Parse the given environment and return useful information about it,
    such as whether it is continuous or not and the size of the action space.

    """
    # Determine whether input is continuous or discrete. Generally, for
    # discrete actions, we will take the softmax of the output
    # probabilities and for the continuous we will use the linear output,
    # rescaled to the action space.
    action_is_continuous = False
    action_low = None
    action_high = None
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    else:
        action_is_continuous = True
        action_low = env.action_space.low
        action_high = env.action_space.high
        action_size = env.action_space.low.shape[0]

    return action_is_continuous, action_size, action_low, action_high
