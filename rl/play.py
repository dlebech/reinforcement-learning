"""Play an environment."""
import collections

import gym
import numpy as np


def play(agent, env_name, num_episodes=5, max_steps=50000, render=True):
    """Use a trained agent in a gym environment.
    
    Parameters
    ----------
    agent
        An instance of an agent.
    env_name
        The name of the environment to play in.
    num_episodes
        Number of episodes to play.
    
    """
    env = gym.make(env_name)
    rewards = [0] * num_episodes
    actions = collections.defaultdict(int)

    episode = 0
    total_steps = 0

    while episode < num_episodes and total_steps < max_steps:
        state = env.reset()
        rewards[episode] = 0
        done = False
        while not done and total_steps < max_steps:
            if render:
                env.render()
            action = agent.act(state, deterministic=True)
            if isinstance(action, np.ndarray):
                action = tuple(round(a, 1) for a in action)
            actions[action] += 1
            state, reward, done, _ = env.step(action)
            rewards[episode] += reward
            total_steps += 1

        print(f"Episode complete with reward {rewards[episode]}")
        episode += 1

    print()
    print(f"Total reward: {sum(rewards)}")
    print(f"Mean game reward: {np.mean(rewards)}")
    print(f"Median game reward: {np.median(rewards)}")
    print(f"Actions taken: {actions}")
