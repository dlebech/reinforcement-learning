"""Asynchronous Advantage Actor Critic (A3C) master agent.

Originally described in paper "Asynchronous Methods for Deep Reinforcement Learning"
https://arxiv.org/pdf/1602.01783.pdf

The code is heavily inspired by a tutorial and accompanying code on tensorflow/models:
https://github.com/tensorflow/models/tree/master/research/a3c_blogpost
The algorithm was rewritten from scratch to gain a more thourough
understanding of the method.

"""
import collections
import logging
import multiprocessing

import gym
import numpy as np
import tensorflow as tf

from rl import constants
from rl.agent import base, util
from rl.agent.a3c import model, actor
from rl.agent.a3c.worker import A3CWorker


logger = logging.getLogger(__name__)


class A3CAgent(base.Agent):
    """Asynchronous Advantage Actor Critic (A3C) master agent."""

    def __init__(
        self,
        env_name,
        save_dir=None,
        max_episodes=constants.DEFAULT_MAX_EPISODES,
        gamma_discount=constants.DEFAULT_GAMMA,
        learning_rate=constants.DEFAULT_LEARNING_RATE,
        thread_count=None,
        worker_update_frequency=constants.DEFAULT_UPDATE_FREQUENCY,
    ):
        super().__init__(env_name, "a3c", save_dir=save_dir)

        self.max_episodes = max_episodes
        self.gamma_discount = gamma_discount
        self.worker_update_frequency = worker_update_frequency
        self.thread_count = thread_count or multiprocessing.cpu_count()

        # Global optimizer and model
        self.optimizer = tf.train.AdamOptimizer(
            use_locking=True, learning_rate=learning_rate
        )
        self.model = model.A3CModel(self.env)

        # Calling the model once will essentially tell it what to expect as
        # input and thus initialize all variables and weights.
        # If we don't do this, the model cannot be updated from worker threads.
        self.model(
            tf.convert_to_tensor(
                np.zeros((1,) + self.env.observation_space.shape), dtype=tf.float32
            )
        )

    def train(self):
        worker_tracker = util.WorkerTracker(self.model, self.model_path)
        workers = []
        for i in range(self.thread_count):
            # Make sure each worker has its own environment. The environment
            # will be unwrapped, meaning that it will potentially never end if
            # the model becomes extremely successful. To avoid this, we set an
            # upper limit to be twice the wrapped maximum episode steps.
            env = self.make_env()
            max_steps = 2 * env.spec.max_episode_steps
            env = env.unwrapped

            # Create and start the worker
            worker = A3CWorker(
                env,
                self.model,
                self.optimizer,
                worker_tracker,
                i + 1,
                self.max_episodes,
                max_steps,
                self.gamma_discount,
                self.worker_update_frequency,
            )
            workers.append(worker)
            worker.start()

        logger.info(f"{len(workers)} workers started.")

        # Wait for all worker to stop running
        logger.info("Waiting for workers to finish.")
        for worker in workers:
            worker.join()
            if worker.exception:
                raise worker.exception

    def act(self, state, deterministic=False):
        return actor.act(self.model, state, deterministic=deterministic)
