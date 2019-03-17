"""Base agent which sets up the target output directory and environment.

"""
import logging
import os
import tempfile

import gym

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, env_name, agent_name, save_dir=None):
        self.env_name = env_name
        self.env = self.make_env()

        if not save_dir:
            save_dir = tempfile.mkdtemp(prefix="tj_rl_")
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.agent_name = agent_name

    @property
    def model_path(self):
        return os.path.join(
            self.save_dir, f"model_{self.agent_name}_{self.env_name}.h5"
        )

    def save_model(self, model):
        logger.info(f"Saving model to: {self.model_path}")
        model.save_weights(self.model_path)

    def load_model(self, model):
        logger.info(f"Loading model from: {self.model_path}")
        model.load_weights(self.model_path)
        return model

    def make_env(self):
        return gym.make(self.env_name)
