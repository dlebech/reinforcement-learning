"""Random agent."""
from queue import Queue

import gym

from rl.agent import base, util


class RandomAgent(base.Agent):
    """Random Agent that will play the specified game

    Parameters
    ----------
    env_name
        Name of the environment to be played
    max_episodes
        Maximum number of episodes to run agent for.

    """

    def __init__(self, env_name, max_episodes):
        super().__init__(env_name, "random")
        self.env = gym.make(env_name)
        self.max_episodes = max_episodes
        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def act(self, state, deterministic=False):
        return self.env.action_space.sample()

    def train(self, render=False):
        reward_avg = 0
        for episode in range(self.max_episodes):
            done = False
            state = self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not done:
                if render:
                    self.env.render()
                # Sample randomly from the action space and step
                state, reward, done, _ = self.env.step(self.act(state))
                steps += 1
                reward_sum += reward
            # Record statistics
            self.global_moving_average_reward = util.record(
                episode,
                reward_sum,
                0,
                self.global_moving_average_reward,
                self.res_queue,
                0,
                steps,
            )

            reward_avg += reward_sum
        final_avg = reward_avg / float(self.max_episodes)
        print(
            "Average score across {} episodes: {}".format(self.max_episodes, final_avg)
        )
        return final_avg
