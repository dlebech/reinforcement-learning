"""Asynchronous Advantage Actor Critic (A3C) worker

"""
import collections
import logging
import threading

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rl import constants
from rl.agent import util
from rl.agent.a3c import model, actor


logger = logging.getLogger(__name__)

# From paper "Asynchronous Methods for Deep Reinforcement Learning"
# Algorithm S3 (in appendix):
#
# assume global policy (P) and value (V) gradients (our global model) grad and grad_v
# initialize thread step counter t
# repeat
#   reset gradients to 0
#   sync thread parameters with global model, grad' = grad and grad_v' = grad_v
#   t_start = t
#   get state s
#
#   repeat
#     perform action a according to policy
#     get reward r and new state s'
#     update local and global step count
#   until
#     s' is terminal or t-t_start = t_max
#
#   R is 0 for a terminal s'
#   otherwise R is V(s, grad_v') (note this is not s')
#
#   for each timestep t in memory
#     R = r_t (timestep reward) + discount * R
#     accumulate gradients grad' and grad_v'
#   update both global grad and grad_v


class A3CWorker(threading.Thread):
    """Asynchronous Advantage Actor Critic (A3C) worker thread."""

    def __init__(
        self,
        env,
        global_model,
        global_optimizer,
        tracker,
        index,
        max_episodes,
        max_steps,
        gamma_discount,
        update_frequency,
    ):
        super().__init__()
        self.env = env
        self.global_model = global_model
        self.global_optimizer = global_optimizer
        self.tracker = tracker
        self.index = index
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.update_frequency = update_frequency
        self.gamma_discount = gamma_discount

        self.model = model.A3CModel(env)

        # TODO: Make this configurable
        self.terminal_reward = -1

    def update_model(self, memory, new_state, terminal):
        # Find total loss
        with tf.GradientTape(persistent=True) as gt:
            total_loss, policy_loss, value_loss = self.calculate_loss(
                memory, new_state, terminal
            )

        value_weights = [
            v for v in self.model.trainable_weights if "value_scope" in v.name
        ]
        actor_weights = [
            v for v in self.model.trainable_weights if "actor_scope" in v.name
        ]

        value_grads = gt.gradient(value_loss, value_weights)
        value_grads, _ = tf.clip_by_global_norm(value_grads, 0.5)
        self.global_optimizer.apply_gradients(
            zip(
                value_grads,
                [
                    v
                    for v in self.global_model.trainable_weights
                    if "value_scope" in v.name
                ],
            )
        )

        actor_grads = gt.gradient(policy_loss, actor_weights)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
        self.global_optimizer.apply_gradients(
            zip(
                actor_grads,
                [
                    v
                    for v in self.global_model.trainable_weights
                    if "actor_scope" in v.name
                ],
            )
        )

        del gt

        # Update local gradients
        # grads = gt.gradient(total_loss, self.model.trainable_weights)
        # grads, _ = tf.clip_by_global_norm(grads, 0.5)

        ## Push to global model
        ## The global optimizer _should_ be thread safe
        # self.global_optimizer.apply_gradients(
        #    zip(grads, self.global_model.trainable_weights)
        # )

        # Fetch global model's weights
        self.model.set_weights(self.global_model.get_weights())

        return total_loss.numpy(), policy_loss.numpy(), value_loss.numpy()

    def calculate_loss(self, memory, new_state, terminal):
        # From algorithm:
        # R is 0 for a terminal s'
        # otherwise R is V(s, grad_v')
        R = 0
        if not terminal:
            _, values = self.model(tf.convert_to_tensor([new_state], dtype=tf.float32))
            R = tf.squeeze(values).numpy()
            assert isinstance(R, np.float32)

        # From algorithm:
        # for each timestep t in memory (backwards)
        #   R = r_t (timestep reward) + discount * R
        #   Update grad' and grad_v'
        # accumulate global gradients based on grad' and grad_v'
        R_list = collections.deque()
        for reward in memory.rewards[::-1]:
            R = reward + self.gamma_discount * R
            # The accumulation of gradients can be accomplished in one call to
            # the model outside of this loop, for efficiency.
            R_list.appendleft(R)

        # With action as a, state as s, simplified we want to get to:
        # grad = log(policy(a))*(R - V(s))
        # grad_v = (R - V(s))^2

        # policy is represented by pi in the algorithm.
        # The basis for the policy(...) and V(...) losses are calculated with a
        # forward step in the model:
        logits, values = self.model(
            tf.convert_to_tensor(memory.states, dtype=tf.float32)
        )

        # R - V(s) (for grad)
        # This is called "advantage" in many tutorials and is a stepping stone
        # on the way to the policy loss.
        advantage = (
            tf.convert_to_tensor(
                np.array(list(R_list)).reshape(values.shape), dtype=tf.float32
            )
            - values
        )

        # (R - V(s))^2 (for grad_v)
        # Called "value loss". It uses the square of the advantage from above.
        value_loss = tf.square(advantage)

        if self.model.action_is_continuous:
            mean, stddev = logits
            mean = actor.rescale_mean(self.model, mean)
            normal_dist = tfp.distributions.Normal(mean, stddev)
            entropy = normal_dist.entropy()
            policy_loss = -normal_dist.log_prob(memory.actions)
        else:
            # Policy loss (really: action loss?)
            # Cross entropy is -sum(policy * log(policy)) which is what we want...
            policy = actor.discrete_action(self.model, logits)
            entropy = tf.losses.softmax_cross_entropy(policy, logits)
            policy_loss = tf.losses.sparse_softmax_cross_entropy(memory.actions, logits)

        # stop_gradient prevents the advantage from contributing to the
        # calculated gradient, "pretending" that it's constant
        policy_loss *= tf.stop_gradient(advantage)

        # total_loss = tf.reduce_mean(value_loss + policy_loss)
        # Use a constant factor to adjust the effect of the value loss and entropy
        total_loss = tf.reduce_mean(policy_loss - 0.01 * entropy + 0.5 * value_loss)

        return (
            total_loss,
            tf.reduce_mean(policy_loss - 0.01 * entropy),
            tf.reduce_mean(value_loss),
        )

    def _run(self):
        memory = WorkerMemory()

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
                self.index, ep_step, ep_reward, ep_loss, ep_loss_policy, ep_loss_value
            )

    def run(self):
        self.exception = None
        try:
            self._run()
        except Exception as e:
            self.exception = e


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
