import argparse
import logging

import tensorflow as tf
tf.enable_eager_execution()

from rl import constants, play
from rl.agent.rnd import RandomAgent
from rl.agent.a3c import A3CAgent
from rl.agent.dqn import DQNAgent


def run():
    parser = argparse.ArgumentParser(
        description="Run a RL agent on an AI gym environment"
    )
    parser.add_argument(
        "--env-name", default=constants.DEFAULT_ENV_NAME, help="The gym environment to run"
    )
    parser.add_argument(
        "--algorithm",
        default=constants.DEFAULT_ALGORITHM,
        type=str,
        choices=['random', 'dqn', 'a3c'],
        help="The algorihtm to use for the RL agent.",
    )
    parser.add_argument(
        "--train", dest="train", action="store_true", help="Train our model."
    )
    parser.add_argument(
        "--lr",
        default=constants.DEFAULT_LEARNING_RATE,
        help="Learning rate for the shared optimizer.",
    )
    parser.add_argument(
        "--update-freq",
        default=constants.DEFAULT_UPDATE_FREQUENCY,
        type=int,
        help="How often to update the global model.",
    )
    parser.add_argument(
        "--max-episodes",
        default=constants.DEFAULT_MAX_EPISODES,
        type=int,
        help="Global maximum number of episodes to run.",
    )
    parser.add_argument(
        "--gamma", default=constants.DEFAULT_GAMMA, help="Discount factor of rewards."
    )
    parser.add_argument(
        "--save-dir", help="Directory in which you desire to save the model."
    )
    parser.add_argument('--log-level', default='DEBUG')

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    if args.algorithm == "random":
        agent = RandomAgent(args.env_name, args.max_episodes)
    elif args.algorithm == "dqn":
        agent = DQNAgent(
            args.env_name,
            max_episodes=args.max_episodes,
            save_dir=args.save_dir
        )
    elif args.algorithm == "a3c":
        agent = A3CAgent(
            env_name=args.env_name,
            #learning_rate=args.lr,
            max_episodes=args.max_episodes,
            save_dir=args.save_dir,
        )

    if args.train:
        agent.train()
    else:
        play.play(agent, args.env_name, args.max_episodes)

if __name__ == "__main__":
    run()