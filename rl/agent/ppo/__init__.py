"""Proximal Policy Optimization

Paper: https://arxiv.org/pdf/1707.06347.pdf

Uses the clipped surrogate objective method for the policy loss.

"""
import collections

import numpy as np
import tensorflow as tf

from rl import constants
from rl.agent import util, actor
from rl.agent.base import Agent
from rl.agent.ppo import model

__all__ = ["PPOAgent"]


class PPOAgent(Agent):
    def __init__(
        self, env_name, save_dir=None, max_episodes=constants.DEFAULT_MAX_EPISODES
    ):
        super().__init__(env_name, "PPO", save_dir=save_dir)
        self.model = model.build_model(self.env)
        self.tracker = util.WorkerTracker(self.model, self.model_path)

        self.max_episodes = max_episodes
        self.gamma_discount = 0.99

        self.clip_error = 0.2
        self.update_frequency = 32
        self.terminal_reward = -1
        self.max_steps = 2 * self.env.spec.max_episode_steps
        self.policy_optimizer = tf.train.AdamOptimizer()
        self.value_optimizer = tf.train.AdamOptimizer()

        self.value_weights = [
            v for v in self.model.trainable_weights if "value_" in v.name
        ]
        self.policy_weights = [
            v for v in self.model.trainable_weights if "policy_" in v.name
        ]

    def train(self):
        self.env = self.env.unwrapped
        memory = util.WorkerMemory()

        while self.tracker.global_episodes < self.max_episodes:
            # Time step is used to keep track of when to update the global model
            # Episode step is used to keep track of the current episode
            time_step = 0
            ep_step = 0
            ep_reward = 0
            ep_loss = 0
            ep_loss_policy = 0
            ep_loss_value = 0

            state = self.env.reset()
            memory.clear()

            while True:
                # Choose an action.
                action = actor.act(self.model, state)
                new_state, reward, done, _ = self.env.step(action)

                # If the environment is done in less than the max steps we're
                # walking, then it's an actually terminal state. We want to
                # make sure we can signal this when updating the model.
                terminal = False
                if done and ep_step < self.max_steps:
                    terminal = True

                    # We also want to make sure to modify the reward for this
                    # action in case we have set a terminal reward. For
                    # example, for the cartpole environment, we will get a +1
                    # reward no matter what, and this is not really beneficial
                    # for learning that terminal is bad :-)
                    if self.terminal_reward:
                        reward = self.terminal_reward

                # Add to memory and step forward
                memory.append(state, action, reward)
                time_step += 1
                ep_step += 1
                ep_reward += reward
                state = new_state

                # Determine if we should update the model
                # We do this at regular intervals...
                if time_step == self.update_frequency or done:
                    tl, pl, vl = self.update_model(memory, new_state, terminal)
                    ep_loss += tl
                    ep_loss_policy += pl
                    ep_loss_value += vl
                    time_step = 0
                    memory.clear()

                if terminal or ep_step >= self.max_steps:
                    # Break for terminal state or if we have stepped too far :-)
                    break

            self.tracker.episode_complete(
                0, ep_step, ep_reward, ep_loss, ep_loss_policy, ep_loss_value
            )

    def update_model(self, memory, new_state, terminal):
        with tf.GradientTape(persistent=True) as gt:
            total_loss, policy_loss, value_loss = self.calculate_loss(
                memory, new_state, terminal
            )

        value_grads = gt.gradient(value_loss, self.value_weights)
        value_grads, _ = tf.clip_by_global_norm(value_grads, 0.5)
        self.value_optimizer.apply_gradients(zip(value_grads, self.value_weights))

        policy_grads = gt.gradient(policy_loss, self.policy_weights)
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, 0.5)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_weights))

        del gt

        return total_loss, policy_loss, value_loss

    def policy_loss(self, y_true, y_pred, advantage):
        # Clipped policy loss from PPO paper (I think...)
        prop_ratio = tf.math.divide(y_true, y_pred)
        clipped = tf.clip_by_value(prop_ratio, 1 - self.clip_error, 1 + self.clip_error)
        l_clip = tf.minimum(prop_ratio * advantage, clipped * advantage)
        return tf.reduce_mean(l_clip)

    def calculate_loss(self, memory, new_state, terminal):
        # See a3c worker agent for more comments on the advantage calculation...
        R = 0
        if not terminal:
            _, values = self.model(tf.convert_to_tensor([new_state], dtype=tf.float32))
            R = tf.squeeze(values).numpy()
            assert isinstance(R, np.float32)

        R_list = collections.deque()
        for reward in memory.rewards[::-1]:
            R = reward + self.gamma_discount * R
            R_list.appendleft(R)

        logits, values = self.model(
            tf.convert_to_tensor(memory.states, dtype=tf.float32)
        )
        advantage = (
            tf.convert_to_tensor(
                np.array(list(R_list)).reshape(values.shape), dtype=tf.float32
            )
            - values
        )

        value_loss = tf.reduce_mean(tf.square(advantage))

        # Only support discrete actions for now...
        logits = tf.nn.softmax(logits)
        policy_loss = self.policy_loss(
            tf.one_hot(memory.actions, logits.shape[1]), logits, advantage
        )

        return policy_loss + 0.5 * value_loss, policy_loss, value_loss

    def act(self, state, deterministic=False):
        return actor.act(self.model, state, deterministic=deterministic)
