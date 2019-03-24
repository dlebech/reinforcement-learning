import collections
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


def record(
    episode,
    episode_reward,
    worker_idx,
    global_ep_reward,
    result_queue,
    total_loss,
    num_steps,
):
    """Helper function to store score and print statistics.

    Parameters
    ----------
    episode
        Current episode
    episode_reward
        Reward accumulated over the current episode
    worker_idx
        Which thread (worker)
    global_ep_reward
        The moving average of the global reward
    result_queue
        Queue storing the moving average of the scores
    total_loss
        The total loss accumualted over the current episode
    num_steps
        The number of steps the episode took to complete

    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    logger.debug(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward


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

        if reward > self.global_best_reward:
            logger.info(
                f"New best reward: {reward} vs. {self.global_best_reward}, "
                f"Saving model to: {self.model_path}"
            )
            with self.lock:
                self.global_model.save_weights(self.model_path)
                self.global_best_reward = reward
